"""Deterministic retrieval by distinct query-token overlap."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ..documents import Document
from ..text import tokenize


@dataclass(frozen=True)
class KeywordHit:
    document_id: str
    score: int


class KeywordIndex:
    def __init__(self, documents: Iterable[Document]) -> None:
        self._postings: dict[str, set[str]] = defaultdict(set)
        for document in documents:
            for token in set(tokenize(f"{document.title} {document.text}")):
                self._postings[token].add(document.document_id)

    def search(self, query: str, top_k: int = 10) -> list[KeywordHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        counts: Counter[str] = Counter()
        for token in set(tokenize(query)):
            counts.update(self._postings.get(token, ()))

        ranked = sorted(counts.items(), key=lambda hit: (-hit[1], hit[0]))
        return [KeywordHit(document_id, score) for document_id, score in ranked[:top_k]]
