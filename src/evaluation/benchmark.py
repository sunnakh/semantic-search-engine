"""Evaluate retrieval baselines on the prepared SciFact validation split."""

import argparse
import json
import sys
import unicodedata
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import NotRequired, TypedDict

from ..data.prepared import DEFAULT_PROCESSED_ROOT, PreparedSplit, load_validation
from ..documents import Document
from ..retrieval.keyword import KeywordIndex
from ..retrieval.lexical import BM25_B, BM25_K1, BM25Index, TfidfIndex
from .metrics import CUTOFF, METRICS, score_ranking

REPORT_ROOT = Path("artifacts/evaluation")
INDEXES: dict[str, type[KeywordIndex | TfidfIndex | BM25Index]] = {
    "keyword": KeywordIndex,
    "tfidf": TfidfIndex,
    "bm25": BM25Index,
}


class QueryResult(TypedDict):
    query_id: str
    ranked_document_ids: list[str]
    relevant_document_ids: list[str]
    metrics: dict[str, float]


class BenchmarkReport(TypedDict):
    dataset_version: str
    input_manifest_sha256: str
    method: str
    method_version: int
    parameters: NotRequired[dict[str, float]]
    model: NotRequired[dict[str, int | str]]
    unicode_data_version: str
    split: str
    document_count: int
    query_count: int
    queries_with_relevant_hit_at_10: int
    metrics: dict[str, float]
    per_query: list[QueryResult]


def dense_rankings(
    documents: list[Document], queries: list[str]
) -> tuple[list[list[str]], dict[str, int | str]]:
    # The neural dependencies are installed only for the dense benchmark.
    from ..retrieval.dense import (
        BATCH_SIZE,
        MAX_SEQUENCE_LENGTH,
        MODEL_ID,
        MODEL_REVISION,
        DenseIndex,
        MiniLMEncoder,
    )

    index = DenseIndex(documents, MiniLMEncoder())
    return index.rank_queries(queries, top_k=CUTOFF), {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cpu",
        "dimensions": index.dimensions,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "similarity": "exact_cosine",
        "sentence_transformers_version": version("sentence-transformers"),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
    }


def benchmark(manifest_path: Path, method: str = "keyword") -> BenchmarkReport:
    data = load_validation(manifest_path)
    queries = sorted(data.queries.items())
    model = None
    if method == "dense":
        rankings, model = dense_rankings(
            data.documents, [query for _, query in queries]
        )
    else:
        index_type = INDEXES.get(method)
        if index_type is None:
            raise ValueError(f"unknown baseline: {method}")
        index = index_type(data.documents)
        rankings = [
            [hit.document_id for hit in index.search(query, top_k=CUTOFF)]
            for _, query in queries
        ]

    report = evaluate_rankings(
        data, rankings, "keyword_overlap" if method == "keyword" else method
    )
    if method == "bm25":
        report["parameters"] = {"k1": BM25_K1, "b": BM25_B}
    if model is not None:
        report["model"] = model
    return report


def evaluate_rankings(
    data: PreparedSplit, rankings: list[list[str]], method: str
) -> BenchmarkReport:
    """Score rankings supplied in ascending validation query ID order."""
    per_query: list[QueryResult] = []
    for query_id, ranked_ids in zip(sorted(data.queries), rankings, strict=True):
        relevant_ids = data.relevant_ids[query_id]
        per_query.append(
            {
                "query_id": query_id,
                "ranked_document_ids": ranked_ids,
                "relevant_document_ids": sorted(relevant_ids),
                "metrics": score_ranking(ranked_ids, relevant_ids),
            }
        )

    return {
        "dataset_version": data.dataset_version,
        "input_manifest_sha256": data.manifest_sha256,
        "method": method,
        "method_version": 1,
        "unicode_data_version": unicodedata.unidata_version,
        "split": "validation",
        "document_count": len(data.documents),
        "query_count": len(per_query),
        "queries_with_relevant_hit_at_10": sum(
            row["metrics"]["recall@10"] > 0 for row in per_query
        ),
        "metrics": {
            name: fmean(row["metrics"][name] for row in per_query) for name in METRICS
        },
        "per_query": per_query,
    }


def write_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".benchmark-", dir=path.parent) as temporary:
        staging = Path(temporary) / "report.json"
        staging.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="prepared SciFact manifest")
    parser.add_argument("--output", type=Path, help="JSON report path")
    parser.add_argument(
        "--method", choices=(*INDEXES, "dense", "all"), default="keyword"
    )
    args = parser.parse_args()
    if args.method == "all" and args.output is not None:
        parser.error("--output requires a single baseline method")
    try:
        manifest = args.manifest
        if manifest is None:
            manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
            if len(manifests) != 1:
                raise ValueError(
                    "run make prepare for a snapshot or select one with --manifest"
                )
            (manifest,) = manifests

        methods = tuple(INDEXES) if args.method == "all" else (args.method,)
        for method in methods:
            report = benchmark(manifest, method)
            output = (
                args.output
                or REPORT_ROOT / report["dataset_version"] / f"{method}-validation.json"
            )
            write_report(report, output)
            print(json.dumps(report["metrics"], sort_keys=True))
            print(output)
    except ImportError as error:
        print(
            f"Dense dependencies unavailable. Run make dense-baseline. {error}",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Baseline benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
