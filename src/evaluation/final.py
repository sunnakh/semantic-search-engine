"""Evaluate a pinned registered model on official SciFact test claims, without fitting."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import TypedDict

import torch

from ..data.integrity import sha256_file
from ..data.prepared import DEFAULT_PROCESSED_ROOT, SPLIT_AUDIT, load_test
from ..registry.tracking import MODEL_NAME
from ..retrieval.dense import (
    MODEL_ID,
    MODEL_REVISION,
    DenseIndex,
    MiniLMEncoder,
    load_minilm,
)
from ..retrieval.lexical import BM25_B, BM25_K1
from ..serving.release import DEFAULT_DENSE_TIMEOUT_MS, load_release
from ..training.candidate import source_provenance
from .benchmark import REPORT_ROOT, BenchmarkReport, dense_rankings, evaluate_rankings
from .metrics import CUTOFF, METRICS


class OverlapSummary(TypedDict):
    audit_sha256: str
    development_test_pair_count: int
    exact_test_query_ids: list[str]
    near_duplicate_test_query_ids: list[str]
    excluded_test_query_ids: list[str]


def save_once(value: object, path: Path) -> None:
    """Publish complete JSON atomically; never replace different existing evidence."""
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".final-", dir=path.parent) as temporary:
        staging = Path(temporary) / "report.json"
        staging.write_bytes(content)
        try:
            os.link(staging, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"refusing to overwrite different evidence: {path}")


def overlap_summary(manifest_path: Path, test_ids: set[str]) -> OverlapSummary:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_path = manifest_path.parent / SPLIT_AUDIT
    if manifest["artifact_sha256"].get(SPLIT_AUDIT) != sha256_file(audit_path):
        raise ValueError("prepared artifact checksum mismatch: split_audit.json")
    pairs = json.loads(audit_path.read_text(encoding="utf-8"))["official_test_matches"]
    exact: set[str] = set()
    near: set[str] = set()
    for pair in pairs:
        ids = test_ids & {pair["left_query_id"], pair["right_query_id"]}
        if len(ids) != 1 or pair["match"] not in ("exact_normalized", "near_duplicate"):
            raise ValueError("invalid official test overlap pair")
        (exact if pair["match"] == "exact_normalized" else near).update(ids)
    return {
        "audit_sha256": sha256_file(audit_path),
        "development_test_pair_count": len(pairs),
        "exact_test_query_ids": sorted(exact),
        "near_duplicate_test_query_ids": sorted(near),
        "excluded_test_query_ids": sorted(exact | near),
    }


def sensitivity_metrics(
    reports: dict[str, BenchmarkReport], excluded: set[str]
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for method, report in reports.items():
        retained = [
            row for row in report["per_query"] if row["query_id"] not in excluded
        ]
        summaries[method] = {
            "query_count": len(retained),
            "metrics": {
                metric: fmean(row["metrics"][metric] for row in retained)
                for metric in METRICS
            }
            if retained
            else None,
        }
    return summaries


def run_test_benchmark(
    manifest_path: Path, tracking_root: Path, model_version: int, output: Path
) -> Path:
    if model_version < 1:
        raise ValueError("select a positive registered model version")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory(prefix="scifact-final-") as temporary:
        # Resolve and checksum the saved, gate-approved encoder before opening test labels.
        release = load_release(
            manifest_path,
            tracking_root,
            model_version,
            DEFAULT_DENSE_TIMEOUT_MS,
            Path(temporary),
        )
        selection = {
            "purpose": "official_test_evaluation_only",
            "release": asdict(release.metadata),
            "registry_model_name": MODEL_NAME,
            "registry_run_id": release.registry_run_id,
            "evaluation_source": source_provenance(),
            "cpu_threads": 4,
            "bm25_parameters": {"k1": BM25_K1, "b": BM25_B},
            "frozen_minilm": {"id": MODEL_ID, "revision": MODEL_REVISION},
        }
        save_once(selection, output / "selection.json")
        data = load_test(manifest_path)
        if data.manifest_sha256 != release.metadata.prepared_manifest_sha256:
            raise ValueError("test inputs changed after selecting the model")
        overlap = overlap_summary(manifest_path, set(data.queries))
        queries = [data.queries[qid] for qid in sorted(data.queries)]
        print(f"Evaluating BM25 on {len(queries)} official test claims", flush=True)
        reports = {
            "bm25": evaluate_rankings(
                data,
                [
                    [
                        hit.document_id
                        for hit in release.bm25.search(query, top_k=CUTOFF)
                    ]
                    for query in queries
                ],
                "bm25",
            )
        }
        reports["bm25"]["parameters"] = {"k1": BM25_K1, "b": BM25_B}
        print("Evaluating frozen MiniLM", flush=True)
        rankings, frozen_model = dense_rankings(data.documents, queries)
        reports["frozen_minilm"] = evaluate_rankings(data, rankings, "dense")
        reports["frozen_minilm"]["model"] = frozen_model
        print(f"Evaluating saved model version {model_version}", flush=True)
        index = DenseIndex.load(
            release.bundle / "index",
            MiniLMEncoder(load_minilm(release.bundle / "encoder")),
        )
        reports["fine_tuned_minilm"] = evaluate_rankings(
            data, index.rank_queries(queries, top_k=CUTOFF), "fine_tuned_minilm"
        )
        reports["fine_tuned_minilm"]["model"] = {
            "candidate_id": release.metadata.candidate_id,
            "model_version": str(model_version),
        }
        report_path = output / "report.json"
        save_once(
            {
                "selection": selection,
                "reports": reports,
                "overlap": overlap,
                "excluding_audited_overlap": sensitivity_metrics(
                    reports, set(overlap["excluded_test_query_ids"])
                ),
            },
            report_path,
        )
        return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tracking-root", type=Path, default=Path("artifacts/mlflow"))
    parser.add_argument(
        "--output", type=Path, help="directory for immutable JSON evidence"
    )
    args = parser.parse_args()
    try:
        manifest = args.manifest
        if manifest is None:
            manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
            if len(manifests) != 1:
                raise ValueError("select one prepared snapshot with --manifest")
            (manifest,) = manifests
        output = (
            args.output or REPORT_ROOT / "official-test" / f"model-{args.model_version}"
        )
        print(
            run_test_benchmark(manifest, args.tracking_root, args.model_version, output)
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(f"Official test benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
