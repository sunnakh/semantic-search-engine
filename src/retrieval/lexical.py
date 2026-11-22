"""TF-IDF cosine and BM25 retrieval over a term-frequency index."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, log, log1p, sqrt

from ..documents import Document
from ..text import tokenize

BM25_K1 = 1.2
BM25_B = 0.75


@dataclass(frozen=True)
class LexicalHit:
    document_id: str
    score: float


class TermIndex:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.lengths: dict[str, int] = {}
        for document in documents:
            counts = Counter(tokenize(f"{document.title} {document.text}"))
            self.lengths[document.document_id] = counts.total()
            for term, frequency in counts.items():
                self.postings[term][document.document_id] = frequency


def top_hits(scores: dict[str, float], top_k: int) -> list[LexicalHit]:
    ranked = sorted(scores.items(), key=lambda hit: (-hit[1], hit[0]))
    return [LexicalHit(document_id, score) for document_id, score in ranked[:top_k]]


class TfidfIndex:
    def __init__(self, documents: Iterable[Document]) -> None:
        self._index = TermIndex(documents)
        size = len(self._index.lengths)
        self._idf: dict[str, float] = {}
        norm_squares: dict[str, float] = defaultdict(float)
        for term, postings in sorted(self._index.postings.items()):
            idf = 1 + log((1 + size) / (1 + len(postings)))
            self._idf[term] = idf
            for document_id, frequency in postings.items():
                norm_squares[document_id] += (frequency * idf) ** 2
        self._norms = {
            document_id: sqrt(value) for document_id, value in norm_squares.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[LexicalHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        counts = Counter(tokenize(query))
        terms = sorted(counts.keys() & self._idf.keys())
        if not terms:
            return []

        weights = {term: counts[term] * self._idf[term] for term in terms}
        query_norm = sqrt(fsum(weight**2 for weight in weights.values()))
        dot_products: dict[str, float] = defaultdict(float)
        for term, weight in weights.items():
            for document_id, frequency in self._index.postings[term].items():
                dot_products[document_id] += weight * (frequency * self._idf[term])

        return top_hits(
            {
                document_id: value / (query_norm * self._norms[document_id])
                for document_id, value in dot_products.items()
            },
            top_k,
        )


class BM25Index:
    def __init__(self, documents: Iterable[Document]) -> None:
        self._index = TermIndex(documents)
        size = len(self._index.lengths)
        total_length = sum(self._index.lengths.values())
        average_length = total_length / size if total_length else 1.0
        self._length_penalty = {
            document_id: BM25_K1 * (1 - BM25_B + BM25_B * (length / average_length))
            for document_id, length in self._index.lengths.items()
        }
        self._idf = {
            term: log1p((size - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._index.postings.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[LexicalHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores: dict[str, float] = defaultdict(float)
        for term in sorted(set(tokenize(query)) & self._idf.keys()):
            idf = self._idf[term]
            for document_id, frequency in self._index.postings[term].items():
                scores[document_id] += (
                    idf
                    * ((BM25_K1 + 1) * frequency)
                    / (frequency + self._length_penalty[document_id])
                )

        return top_hits(scores, top_k)
