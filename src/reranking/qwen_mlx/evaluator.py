"""Qwen reranker evaluation and report generation."""

import importlib
import json
import sys
from math import isfinite
from pathlib import Path
from typing import cast

from ...data.integrity import sha256_file
from ...data.prepared import load_validation
from ...evaluation.benchmark import BenchmarkReport, evaluate_rankings, write_report
from ...evaluation.gate import evaluate_ranking_gate, verified_metrics
from ...evaluation.metrics import CUTOFF
from ...retrieval.lexical import BM25Index
from .checkpoint import PROMPT_VERSION, checkpoint_hashes
from .scorer import RelevanceScorer


def evaluate(
    prepared: Path,
    run: Path,
    model_path: Path | None,
    top_n: int,
    query_limit: int | None,
    candidate_report: Path | None = None,
) -> Path:
    mlx_lm = importlib.import_module("mlx_lm")
    tuner_utils = importlib.import_module("mlx_lm.tuner.utils")
    load = mlx_lm.load
    load_adapters = tuner_utils.load_adapters

    data = load_validation(prepared)
    candidate_rows: dict[str, list[str]] = {}
    if candidate_report is not None:
        if top_n != CUTOFF:
            raise ValueError(
                f"the Qwen embedding report uses a fixed top-{CUTOFF} cutoff"
            )
        _, source = verified_metrics(
            candidate_report, data, "frozen_qwen3_embedding_0_6b"
        )
        candidate_rows = {
            row["query_id"]: row["ranked_document_ids"] for row in source["per_query"]
        }
        if any(len(set(row)) != len(row) for row in candidate_rows.values()):
            raise ValueError("the candidate report contains duplicate document IDs")

    manifest = json.loads(run.read_text(encoding="utf-8"))
    adapter = run.parent / "adapter"
    model_path = (
        (model_path or Path(manifest["base_model_path"])).expanduser().resolve()
    )
    if (
        manifest["input_manifest_sha256"] != data.manifest_sha256
        or manifest["prompt_version"] != PROMPT_VERSION
        or manifest["base_model_sha256"] != checkpoint_hashes(model_path)
        or any(
            sha256_file(adapter / name) != manifest["adapter_sha256"][name]
            for name in ("adapters.safetensors", "adapter_config.json")
        )
    ):
        raise ValueError(
            "the prepared data, checkpoint, or adapter does not match the run"
        )
    if query_limit is not None:
        ids = sorted(data.queries)[:query_limit]
        data.queries = {key: data.queries[key] for key in ids}
        data.relevant_ids = {key: data.relevant_ids[key] for key in ids}

    documents = {paper.document_id: paper for paper in data.documents}
    query_ids = sorted(data.queries)
    queries = [data.queries[key] for key in query_ids]
    if candidate_report is None:
        index = BM25Index(data.documents)
        candidates = [
            [hit.document_id for hit in index.search(query, top_k=top_n)]
            for query in queries
        ]
        baseline = "bm25"
    else:
        candidates = [candidate_rows[key] for key in query_ids]
        baseline = "qwen_embedding"
    loaded = load(str(model_path))
    model, tokenizer = loaded[0], loaded[1]
    model.eval()
    scorer = RelevanceScorer(tokenizer)

    def rerank(label: str) -> list[list[str]]:
        rankings = []
        for number, (query, document_ids) in enumerate(
            zip(queries, candidates, strict=True), 1
        ):
            scores = []
            for document_id in document_ids:
                logits = scorer.label_logits(
                    model, scorer.tokens(query, documents[document_id])
                )
                confidence = cast(float, (logits[1] - logits[0]).item())
                if not isfinite(confidence):
                    raise RuntimeError("Qwen reranker returned a non-finite score")
                scores.append(confidence)
            order = sorted(
                range(len(document_ids)), key=lambda index: (-scores[index], index)
            )
            rankings.append([document_ids[index] for index in order[:CUTOFF]])
            if number == 1 or number % 10 == 0 or number == len(queries):
                print(
                    f"{label} · {number}/{len(queries)} claims",
                    file=sys.stderr,
                    flush=True,
                )
        return rankings

    reports: dict[str, BenchmarkReport] = {
        baseline: evaluate_rankings(
            data, [row[:CUTOFF] for row in candidates], baseline
        )
    }
    if candidate_report is None:
        frozen = rerank("Frozen Qwen")
        reports["frozen_qwen"] = evaluate_rankings(
            data, frozen, "frozen_qwen35_9b_reranker"
        )

    model = load_adapters(model, str(adapter))
    model.eval()
    tuned = rerank("QLoRA Qwen")
    reports["qlora_qwen"] = evaluate_rankings(data, tuned, "qlora_qwen35_9b_reranker")
    suffix = (
        "validation" if query_limit is None else f"validation-sample-{len(queries)}"
    )
    if candidate_report is not None:
        suffix = f"qwen-embedding-{suffix}"
    for name, report in reports.items():
        write_report(report, run.parent / f"{name}-{suffix}.json")
    comparison = run.parent / f"comparison-{suffix}.json"
    result: dict[str, object] = {
        "validation_queries": len(queries),
        "requested_subset": query_limit,
        "candidate_depth": top_n,
        **{method: report["metrics"] for method, report in reports.items()},
    }
    if query_limit is None:
        candidate_report_path = run.parent / f"qlora_qwen-{suffix}.json"
        baseline_report_path = run.parent / f"{baseline}-{suffix}.json"
        if candidate_report is None:
            reference_report_path = run.parent / f"frozen_qwen-{suffix}.json"
            decision = evaluate_ranking_gate(
                candidate_report_path,
                data,
                "qlora_qwen35_9b_reranker",
                baseline_report_path,
                baseline,
                reference_report_path,
                "frozen_qwen35_9b_reranker",
            )
        else:
            decision = evaluate_ranking_gate(
                candidate_report_path,
                data,
                "qlora_qwen35_9b_reranker",
                baseline_report_path,
                baseline,
            )
        result["gate"] = {
            "passed": decision.passed,
            "reference": decision.reference,
            "floors": decision.floors,
            "reasons": list(decision.reasons),
        }
    if candidate_report is not None:
        result["candidate_report_sha256"] = sha256_file(candidate_report)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    comparison.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return comparison
