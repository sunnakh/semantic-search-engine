"""Ranking metrics for SciFact's binary relevance judgments."""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from math import fsum, log2

METRICS = ("ndcg@10", "recall@5", "recall@10", "mrr@10")
CUTOFF = 10
DISCOUNTS = tuple(1 / log2(rank + 1) for rank in range(1, CUTOFF + 1))


def score_ranking(
    ranked_ids: Sequence[str], relevant_ids: AbstractSet[str]
) -> dict[str, float]:
    if not relevant_ids:
        raise ValueError("ranking metrics require at least one relevant document")

    top = ranked_ids[:CUTOFF]
    if len(set(top)) != len(top):
        raise ValueError("ranked document IDs must be unique")

    hits = [
        rank for rank, document_id in enumerate(top, 1) if document_id in relevant_ids
    ]
    ideal = fsum(DISCOUNTS[: len(relevant_ids)])
    return {
        "ndcg@10": fsum(DISCOUNTS[rank - 1] for rank in hits) / ideal,
        "recall@5": sum(rank <= 5 for rank in hits) / len(relevant_ids),
        "recall@10": len(hits) / len(relevant_ids),
        "mrr@10": 1 / hits[0] if hits else 0.0,
    }
