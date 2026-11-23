import unittest
from importlib.util import find_spec
from unittest.mock import patch

from src.data.prepared import PreparedSplit
from src.documents import Document

MONITORING_AVAILABLE = all(
    find_spec(name) for name in ("numpy", "prometheus_client", "pydantic")
)
if MONITORING_AVAILABLE:
    from src.monitoring.sentinel import QualityMetrics, Replay, replay
    from src.serving.schemas import SearchHit, SearchResponse


@unittest.skipUnless(MONITORING_AVAILABLE, "dense and serving extras required")
class SentinelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = PreparedSplit(
            "dataset",
            "manifest",
            [Document("d1", "", ""), Document("d2", "", "")],
            {"b": "Second claim", "a": "First claim"},
            {"a": {"d1"}, "b": {"d2"}},
        )

    @staticmethod
    def response(
        document_id: str, *, release_id: str = "release", fallback: bool = False
    ) -> "SearchResponse":
        return SearchResponse(
            method="bm25" if fallback else "dense",
            model_version="1",
            release_id=release_id,
            request_id="request",
            fallback=fallback,
            results=[
                SearchHit(document_id=document_id, title="", abstract="", score=1)
            ],
            latency_ms=1,
        )

    def test_scores_effective_ranking_and_counts_fallback(self) -> None:
        with (
            patch("src.monitoring.sentinel.release_for", return_value="release"),
            patch(
                "src.monitoring.sentinel.search",
                side_effect=[self.response("d1"), self.response("d1", fallback=True)],
            ) as search,
        ):
            result = replay("http://example.test", self.data)
        self.assertEqual(result.scores["ndcg@10"], 0.5)
        self.assertEqual(result.scores["recall@5"], 0.5)
        self.assertEqual(result.fallback_share, 0.5)
        self.assertEqual(
            [call.args[1].query for call in search.call_args_list],
            ["First claim", "Second claim"],
        )

    def test_rejects_unknown_documents_and_mixed_releases(self) -> None:
        for response, message in (
            (self.response("unknown"), "invalid sentinel ranking"),
            (self.response("d1", release_id="other"), "release changed"),
        ):
            with (
                self.subTest(message=message),
                patch("src.monitoring.sentinel.release_for", return_value="release"),
                patch("src.monitoring.sentinel.search", return_value=response),
                self.assertRaisesRegex(ValueError, message),
            ):
                replay("http://example.test", self.data)

    def test_failed_replay_withdraws_partial_quality_metrics(self) -> None:
        metrics = QualityMetrics({"ndcg@10": 0.7, "recall@5": 0.8})
        metrics.record(Replay("release", {"ndcg@10": 0.7, "recall@5": 0.8}, 0))
        succeeded = {family.name: family.samples for family in metrics.collect()}
        metrics.record(None)
        failed = {family.name: family.samples for family in metrics.collect()}
        self.assertIn("scifact_sentinel_score", succeeded)
        self.assertNotIn("scifact_sentinel_score", failed)
        self.assertEqual(failed["scifact_sentinel_run_ok"][0].value, 0)
        self.assertEqual(
            failed["scifact_sentinel_last_success_timestamp_seconds"],
            succeeded["scifact_sentinel_last_success_timestamp_seconds"],
        )
