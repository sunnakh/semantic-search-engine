"""Deterministic training-example construction for the Qwen reranker."""

import random
from collections.abc import Iterator

from ...data.prepared import PreparedSplit
from ...documents import Document
from ...retrieval.lexical import BM25Index
from .checkpoint import SEED


def training_groups(data: PreparedSplit) -> list[tuple[str, Document, Document]]:
    documents = {document.document_id: document for document in data.documents}
    index = BM25Index(data.documents)
    excluded = set(data.exact_test_overlap_ids)
    groups = []
    for query_id in sorted(data.queries.keys() - excluded):
        query = data.queries[query_id]
        relevant = data.relevant_ids[query_id]
        negatives = [
            hit.document_id
            for hit in index.search(query, top_k=50)
            if hit.document_id not in relevant
        ]
        if negatives:
            for offset, document_id in enumerate(sorted(relevant)):
                groups.append(
                    (
                        query,
                        documents[document_id],
                        documents[negatives[offset % len(negatives)]],
                    )
                )
    if not groups:
        raise ValueError("no SciFact training claims with BM25 negatives")
    return groups


def training_examples(
    groups: list[tuple[str, Document, Document]],
) -> Iterator[tuple[str, Document, int]]:
    rng = random.Random(SEED)
    while True:
        rng.shuffle(groups)
        for query, positive, negative in groups:
            yield query, positive, 1
            yield query, negative, 0
