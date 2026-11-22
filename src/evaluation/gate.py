"""Verify a candidate bundle and apply the offline registration policy."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.integrity import file_hashes, manifest_digest
from ..data.prepared import PreparedSplit
from .benchmark import evaluate_rankings
from .metrics import CUTOFF

RECALL_TOLERANCE = 0.02
POLICY_VERSION = 1
ENCODER_FILES = {
    "encoder/model.safetensors",
    "encoder/modules.json",
    "encoder/tokenizer.json",
    "encoder/1_Pooling/config.json",
    "index/document_ids.json",
    "index/vectors.npy",
}
REPORT_FILES = {
    "candidate": "candidate-validation.json",
    "bm25": "bm25-validation.json",
    "frozen_minilm": "frozen_minilm-validation.json",
}


@dataclass(frozen=True)
class GateDecision:
    reference: str
    metrics: dict[str, dict[str, float]]
    floors: dict[str, float]
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def check_bundle(manifest_path: Path, data: PreparedSplit) -> dict[str, Any]:
    manifest = read_object(manifest_path)
    if manifest.get("status") != "evaluated" or manifest.get("manifest_version") != 1:
        raise ValueError("the gate requires an evaluated candidate manifest")
    if (
        manifest.get("dataset_version") != data.dataset_version
        or manifest.get("input_manifest_sha256") != data.manifest_sha256
    ):
        raise ValueError(
            "candidate and validation inputs must use the same prepared manifest"
        )

    metadata = {"candidate_id", "status", "validation_metrics", "artifact_sha256"}
    lineage = {key: value for key, value in manifest.items() if key not in metadata}
    if manifest.get("candidate_id") != manifest_digest(lineage):
        raise ValueError("candidate ID does not match its recorded lineage")

    bundle = manifest_path.parent
    if manifest_path.is_symlink() or any(
        path.is_symlink() for path in bundle.rglob("*")
    ):
        raise ValueError("candidate bundles must contain regular files and directories")
    hashes = file_hashes(bundle)
    hashes.pop(manifest_path.name)
    required = ENCODER_FILES | set(REPORT_FILES.values())
    if not required <= hashes.keys() or hashes != manifest.get("artifact_sha256"):
        raise ValueError("candidate artifact checksum mismatch or missing files")
    model_index = {
        key: value
        for key, value in hashes.items()
        if key.startswith(("encoder/", "index/"))
    }
    if model_index != manifest.get("model_and_index_sha256"):
        raise ValueError(
            "candidate model and index hashes do not match the manifest lineage"
        )

    document_ids = json.loads(
        (bundle / "index/document_ids.json").read_text(encoding="utf-8")
    )
    if document_ids != sorted(document.document_id for document in data.documents):
        raise ValueError(
            "candidate index does not cover the prepared corpus in document ID order"
        )
    inference = manifest.get("inference", {})
    dimensions = inference.get("dimensions") if isinstance(inference, dict) else None
    pooling = read_object(bundle / "encoder/1_Pooling/config.json")
    if (
        type(dimensions) is not int
        or dimensions < 1
        or dimensions
        != pooling.get("embedding_dimension", pooling.get("word_embedding_dimension"))
    ):
        raise ValueError("candidate index dimension does not match the encoder")
    vectors = np.load(bundle / "index/vectors.npy", mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.shape != (len(document_ids), dimensions)
        or vectors.dtype != np.float32
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
    ):
        raise ValueError(
            "candidate index must contain one normalized float32 vector per document"
        )
    return manifest


def verified_metrics(
    path: Path,
    data: PreparedSplit,
    method: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    report = read_object(path)
    if (
        report.get("dataset_version") != data.dataset_version
        or report.get("input_manifest_sha256") != data.manifest_sha256
        or report.get("split") != "validation"
        or report.get("method") != method
        or report.get("query_count") != len(data.queries)
        or report.get("document_count") != len(data.documents)
    ):
        raise ValueError(f"incomparable validation report: {path.name}")

    rows = report.get("per_query")
    if not isinstance(rows, list) or len(rows) != len(data.queries):
        raise ValueError(f"incomplete per-query rankings: {path.name}")
    corpus_ids = {document.document_id for document in data.documents}
    rankings: list[list[str]] = []
    for query_id, row in zip(sorted(data.queries), rows, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("query_id") != query_id
            or row.get("relevant_document_ids") != sorted(data.relevant_ids[query_id])
        ):
            raise ValueError(f"validation queries or judgments differ: {path.name}")
        ranking = row.get("ranked_document_ids")
        if (
            not isinstance(ranking, list)
            or len(ranking) > CUTOFF
            or any(
                not isinstance(value, str) or value not in corpus_ids
                for value in ranking
            )
        ):
            raise ValueError(f"invalid ranking for query {query_id}: {path.name}")
        rankings.append(ranking)

    recomputed = evaluate_rankings(data, rankings, method)
    if (
        report.get("metrics") != recomputed["metrics"]
        or rows != recomputed["per_query"]
    ):
        raise ValueError(f"reported metrics do not match rankings: {path.name}")
    return recomputed["metrics"], report


def evaluate_candidate(
    manifest_path: Path,
    data: PreparedSplit,
    manifest: dict[str, Any],
    champion_report: Path | None = None,
) -> GateDecision:
    bundle = manifest_path.parent
    candidate, report = verified_metrics(
        bundle / REPORT_FILES["candidate"], data, "fine_tuned_minilm"
    )
    model = report.get("model", {})
    if (
        not isinstance(model, dict)
        or model.get("candidate_id") != manifest["candidate_id"]
    ):
        raise ValueError("candidate validation report references a different bundle")
    bm25, _ = verified_metrics(bundle / REPORT_FILES["bm25"], data, "bm25")
    frozen, frozen_report = verified_metrics(
        bundle / REPORT_FILES["frozen_minilm"], data, "dense"
    )
    base_model = manifest.get("base_model", {})
    frozen_model = frozen_report.get("model", {})
    if (
        not isinstance(base_model, dict)
        or not isinstance(frozen_model, dict)
        or any(
            base_model.get(key) != frozen_model.get(key) for key in ("id", "revision")
        )
    ):
        raise ValueError("frozen reference report does not match the base model")

    metrics = {"candidate": candidate, "bm25": bm25, "frozen_minilm": frozen}
    if metrics != manifest.get("validation_metrics"):
        raise ValueError("validation metrics do not match the candidate manifest")
    reference = "frozen_minilm"
    if champion_report is not None:
        metrics["champion"], _ = verified_metrics(
            champion_report, data, "fine_tuned_minilm"
        )
        reference = "champion"

    floors = {
        "ndcg@10": max(bm25["ndcg@10"], metrics[reference]["ndcg@10"]),
        "recall@5": max(
            0.0,
            max(bm25["recall@5"], metrics[reference]["recall@5"]) - RECALL_TOLERANCE,
        ),
    }
    reasons = tuple(
        f"{name} {candidate[name]:.5f} below {floor:.5f}"
        for name, floor in floors.items()
        if candidate[name] < floor
    )
    return GateDecision(reference, metrics, floors, reasons)
