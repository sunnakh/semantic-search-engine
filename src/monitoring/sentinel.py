"""Replay labeled SciFact queries against a running search release."""

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from urllib.request import urlopen

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily

from ..data.prepared import PreparedSplit, load_validation
from ..evaluation.benchmark import evaluate_rankings
from ..evaluation.gate import verified_metrics
from ..evaluation.metrics import CUTOFF
from ..serving.client import DEFAULT_API_URL, SearchError, search
from ..serving.schemas import SearchRequest

METRIC_LABELS = {"ndcg@10": "ndcg_at_10", "recall@5": "recall_at_5"}
TOLERANCE = 0.02


@dataclass(frozen=True)
class Replay:
    release_id: str
    scores: dict[str, float]
    fallback_share: float


def release_for(api_url: str, data: PreparedSplit) -> str:
    with urlopen(api_url.rstrip("/") + "/api/v1/models", timeout=5) as response:
        metadata = json.load(response)
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset_version") != data.dataset_version
        or metadata.get("prepared_manifest_sha256") != data.manifest_sha256
    ):
        raise ValueError("the served corpus does not match the sentinel snapshot")
    release_id = metadata.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise ValueError("the API did not report a release ID")
    return release_id


def replay(api_url: str, data: PreparedSplit) -> Replay:
    release_id = release_for(api_url, data)
    corpus_ids = {doc.document_id for doc in data.documents}
    rankings = []
    fallbacks = 0
    for query_id in sorted(data.queries):
        response = search(
            api_url, SearchRequest(query=data.queries[query_id], top_k=CUTOFF)
        )
        if response.release_id != release_id:
            raise ValueError("the served release changed during sentinel replay")
        if response.method != ("bm25" if response.fallback else "dense"):
            raise ValueError("the API returned an unexpected retrieval method")
        ranked_ids = [result.document_id for result in response.results]
        if len(ranked_ids) > CUTOFF or not corpus_ids.issuperset(ranked_ids):
            raise ValueError("the API returned an invalid sentinel ranking")
        rankings.append(ranked_ids)
        fallbacks += response.fallback
    if release_for(api_url, data) != release_id:
        raise ValueError("the served release changed during sentinel replay")
    report = evaluate_rankings(data, rankings, "served_dense")
    return Replay(release_id, report["metrics"], fallbacks / len(rankings))


class QualityMetrics:
    def __init__(self, reference: dict[str, float]) -> None:
        self.floors = {
            name: max(0.0, reference[name] - TOLERANCE) for name in METRIC_LABELS
        }
        self.state: tuple[Replay | None, float] = (None, 0.0)

    def record(self, result: Replay | None) -> None:
        self.state = (result, time() if result is not None else self.state[1])

    def collect(self) -> Iterator[GaugeMetricFamily]:
        result, last_success = self.state
        yield GaugeMetricFamily(
            "scifact_sentinel_run_ok",
            "Whether the latest sentinel replay completed successfully",
            value=int(result is not None),
        )
        yield GaugeMetricFamily(
            "scifact_sentinel_last_success_timestamp_seconds",
            "Unix timestamp of the latest successful replay, or zero before success",
            value=last_success,
        )
        floors = GaugeMetricFamily(
            "scifact_sentinel_floor", "Approved quality floor", labels=["metric"]
        )
        for name, label in METRIC_LABELS.items():
            floors.add_metric([label], self.floors[name])
        yield floors
        if result is None:
            return
        scores = GaugeMetricFamily(
            "scifact_sentinel_score",
            "Latest served SciFact quality score",
            labels=["metric"],
        )
        for name, label in METRIC_LABELS.items():
            scores.add_metric([label], result.scores[name])
        yield scores
        yield GaugeMetricFamily(
            "scifact_sentinel_fallback_share",
            "Fraction of sentinel queries answered by BM25 fallback",
            value=result.fallback_share,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--port", type=int, default=9101)
    args = parser.parse_args()
    if args.interval_seconds < 1 or not 1 <= args.port <= 65535:
        parser.error("interval must be positive and port must be in 1–65535")
    data = load_validation(args.manifest)
    reference, _ = verified_metrics(args.reference_report, data, "fine_tuned_minilm")
    metrics = QualityMetrics(reference)
    registry = CollectorRegistry()
    registry.register(metrics)
    server, _ = start_http_server(args.port, addr="127.0.0.1", registry=registry)
    try:
        while True:
            try:
                result = replay(args.api_url, data)
            except (OSError, SearchError, ValueError) as error:
                metrics.record(None)
                print(f"Sentinel replay failed: {error}", file=sys.stderr, flush=True)
            else:
                metrics.record(result)
                print(
                    json.dumps(
                        {
                            "release_id": result.release_id,
                            "dataset_version": data.dataset_version,
                            "queries": len(data.queries),
                            "metrics": result.scores,
                            "fallback_share": result.fallback_share,
                        }
                    ),
                    flush=True,
                )
            sleep(args.interval_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
