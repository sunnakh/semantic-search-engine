import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

UI_AVAILABLE = all(find_spec(name) for name in ("pydantic", "streamlit"))
if UI_AVAILABLE:
    from streamlit.testing.v1 import AppTest

    from src.serving.schemas import SearchHit, SearchRequest, SearchResponse
    from src.ui.client import SearchError

APP = Path(__file__).resolve().parents[1] / "src/ui/app.py"


@unittest.skipUnless(UI_AVAILABLE, "ui extra required")
class SearchUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppTest.from_file(str(APP)).run()

    def submit(self, query: str, mode: str | None = None) -> None:
        self.app.text_input[0].set_value(query)
        if mode is not None:
            self.app.selectbox[0].select(mode)
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)

    @staticmethod
    def response(
        request: "SearchRequest", *, fallback: bool = False, empty: bool = False
    ) -> "SearchResponse":
        return SearchResponse(
            method="bm25" if fallback else request.method,
            model_version="3",
            release_id="release-1",
            request_id="request-1",
            fallback=fallback,
            latency_ms=12.4,
            results=[]
            if empty
            else [
                SearchHit(
                    document_id="123",
                    title="Research paper",
                    abstract="Abstract",
                    score=0.6,
                )
            ],
        )

    def test_compare_calls_dense_and_bm25_with_same_query(self) -> None:
        with patch(
            "src.ui.client.search",
            side_effect=lambda url, request: self.response(request),
        ) as search:
            self.submit("  heart health  ")
        requests = [call.args[1] for call in search.call_args_list]
        self.assertCountEqual(
            [request.method for request in requests], ["dense", "bm25"]
        )
        self.assertTrue(all(request.query == "heart health" for request in requests))
        self.assertEqual(
            [heading.value for heading in self.app.subheader], ["Dense", "BM25"]
        )
        self.assertEqual(len(self.app.json), 2)

    def test_fallback_uses_actual_score_label(self) -> None:
        with patch(
            "src.ui.client.search",
            side_effect=lambda url, request: self.response(request, fallback=True),
        ):
            self.submit("heart health", "Dense")
        self.assertIn("Showing BM25 results", self.app.warning[0].value)
        self.assertTrue(
            any("BM25 score" in element.proto.body for element in self.app.get("html"))
        )

    def test_empty_results_and_partial_failure_stay_visible(self) -> None:
        def search(api_url: str, request: "SearchRequest") -> "SearchResponse":
            if request.method == "dense":
                raise SearchError("The search API timed out.")
            return self.response(request, empty=True)

        with patch("src.ui.client.search", side_effect=search):
            self.submit("heart health")
        self.assertIn("timed out", self.app.error[0].value)
        self.assertIn("No papers matched", self.app.info[0].value)

    def test_blank_query_does_not_call_api(self) -> None:
        with patch("src.ui.client.search") as search:
            self.submit("  ")
        search.assert_not_called()
        self.assertIn("Enter a research query", self.app.warning[0].value)
