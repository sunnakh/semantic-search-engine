"""Browse and compare ranked SciFact papers in Streamlit."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import escape

import streamlit as st

from src.serving.schemas import (
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    Method,
    SearchRequest,
    SearchResponse,
)
from src.ui import client

METHODS: dict[Method, str] = {"dense": "Dense", "bm25": "BM25", "keyword": "Keyword"}
SEARCH_MODES: dict[str, tuple[Method, ...]] = {
    "Compare dense and BM25": ("dense", "bm25"),
    "Dense": ("dense",),
    "BM25": ("bm25",),
    "Keyword": ("keyword",),
}
SCORE_LABELS: dict[Method, str] = {
    "dense": "Cosine similarity",
    "bm25": "BM25 score",
    "keyword": "Matching terms",
}

STYLES = """
<style>
.stMainBlockContainer { max-width: 1170px; padding-top: 2.8rem; }
.scifact-header {
    padding: 1.15rem 0 1.5rem;
    border-bottom: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
    margin-bottom: 1rem;
}
.scifact-header h1 {
    font: 600 clamp(2.1rem, 5vw, 3.7rem)/1.06 Georgia, serif;
    letter-spacing: -0.045em;
    margin: 0 0 .6rem;
    padding: 0;
}
.scifact-header p { margin: 0; opacity: .75; font-size: 1.03rem; }
.st-key-search-form button[kind="primaryFormSubmit"] {
    background: #17685d;
    border-color: #17685d;
    color: #fff;
    font-weight: 600;
}
.st-key-search-form button[kind="primaryFormSubmit"]:hover { background: #105349; }
.scifact-summary {
    font-size: .91rem;
    margin: .6rem 0;
    overflow-wrap: anywhere;
}
.scifact-paper {
    border-top: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
    padding-top: .7rem;
}
.scifact-paper h3 {
    font: 600 1.15rem/1.3 Georgia, serif;
    letter-spacing: -.01em;
    padding: 0;
    margin: 0 0 .32rem;
    overflow-wrap: anywhere;
}
.scifact-paper p { opacity: .77; margin: 0; font-size: .85rem; }
</style>
"""


@dataclass(frozen=True)
class SearchOutcome:
    method: Method
    response: SearchResponse | None = None
    error: str | None = None


def run_searches(
    api_url: str, query: str, methods: tuple[Method, ...], top_k: int
) -> list[SearchOutcome]:
    def run(method: Method) -> SearchOutcome:
        try:
            payload = SearchRequest(query=query, method=method, top_k=top_k)
            return SearchOutcome(method, response=client.search(api_url, payload))
        except client.SearchError as error:
            return SearchOutcome(method, error=str(error))

    if len(methods) == 1:
        return [run(methods[0])]
    with ThreadPoolExecutor(max_workers=len(methods)) as workers:
        return list(workers.map(run, methods))


def show_results(outcome: SearchOutcome) -> None:
    st.subheader(METHODS[outcome.method])
    if outcome.error is not None:
        st.error(outcome.error)
        return
    response = outcome.response
    assert response is not None
    if response.fallback:
        st.warning(
            f"{METHODS[outcome.method]} was unavailable. Showing {METHODS[response.method]} results."
        )

    count = len(response.results)
    st.caption(
        f"{count} {'paper' if count == 1 else 'papers'} · {response.latency_ms:.1f} ms server time"
    )
    if not response.results:
        st.info("No papers matched. Try a broader query or a different method.")

    for rank, paper in enumerate(response.results, start=1):
        title = escape(paper.title or "Untitled paper")
        score = (
            f"{paper.score:.0f}"
            if response.method == "keyword"
            else f"{paper.score:.3f}"
        )
        st.html(
            '<div class="scifact-paper">'
            f"<h3>{rank}. {title}</h3>"
            f"<p>{SCORE_LABELS[response.method]} {score} · SciFact ID {escape(paper.document_id)}</p>"
            "</div>"
        )
        if paper.abstract:
            with st.expander("Read abstract"):
                st.html(
                    f'<p style="white-space: pre-line">{escape(paper.abstract)}</p>'
                )

    with st.expander("Request details"):
        st.json(
            {
                "method": response.method,
                "model_version": response.model_version,
                "release_id": response.release_id,
                "request_id": response.request_id,
                "fallback": response.fallback,
            },
            expanded=False,
        )


def main() -> None:
    st.set_page_config(page_title="SciFact Search", layout="wide")
    st.html(STYLES)
    st.html(
        '<header class="scifact-header"><h1>SciFact Search</h1>'
        "<p>Find scientific abstracts and compare how retrieval methods rank them.</p></header>"
    )
    with st.form("search-form", border=False):
        query = st.text_input(
            "Research query",
            placeholder="exercise and heart health",
            max_chars=MAX_QUERY_LENGTH,
        )
        mode_column, count_column, submit_column = st.columns(
            [4, 2, 1], vertical_alignment="bottom"
        )
        with mode_column:
            mode = st.selectbox("Retrieval method", list(SEARCH_MODES))
        with count_column:
            top_k = st.number_input(
                "Papers per method", min_value=1, max_value=MAX_TOP_K, value=5
            )
        with submit_column:
            submitted = st.form_submit_button("Search", type="primary", width="stretch")

    if submitted:
        st.session_state.pop("search", None)
        query = query.strip()
        if not query:
            st.warning("Enter a research query.")
            return
        api_url = os.environ.get("SCIFACT_API_URL", client.DEFAULT_API_URL)
        with st.spinner("Searching SciFact…"):
            results = run_searches(api_url, query, SEARCH_MODES[mode], top_k)
        st.session_state["search"] = (query, results)

    if "search" not in st.session_state:
        return
    query, results = st.session_state["search"]
    st.html(
        f'<p class="scifact-summary">Results for <strong>{escape(query)}</strong></p>'
    )
    if len(results) == 1:
        show_results(results[0])
        return
    release_ids = {
        item.response.release_id for item in results if item.response is not None
    }
    if len(release_ids) > 1:
        st.warning(
            "The results came from different releases. Search again for a direct comparison."
        )
    st.caption("Compare the rankings. Each retrieval method uses its own score scale.")
    for column, result in zip(st.columns(len(results)), results, strict=True):
        with column:
            show_results(result)


if __name__ == "__main__":
    main()
