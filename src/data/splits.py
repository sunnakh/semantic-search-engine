"""Deterministic query splits and a lexical overlap audit for SciFact."""

import hashlib
from itertools import accumulate, combinations, pairwise
from typing import TypedDict

from ..text import tokenize

SPLIT_POLICY = {
    "seed": 2026,
    "validation_percent": 20,
    "bigram_jaccard_percent": 80,
}
SPLITS = ("train", "validation", "test")


class ClaimMatch(TypedDict):
    left_query_id: str
    right_query_id: str
    match: str
    similarity: float


def audit_claims(queries: dict[str, str], test_ids: set[str]) -> list[ClaimMatch]:
    tokens = {query_id: tokenize(text) for query_id, text in queries.items()}
    bigrams = {query_id: set(pairwise(words)) for query_id, words in tokens.items()}
    matches: list[ClaimMatch] = []
    for left, right in combinations(sorted(queries), 2):
        if left in test_ids and right in test_ids:
            continue

        if tokens[left] and tokens[left] == tokens[right]:
            match, similarity = "exact_normalized", 1.0
        else:
            intersection = len(bigrams[left] & bigrams[right])
            union = len(bigrams[left]) + len(bigrams[right]) - intersection
            if (
                not union
                or 100 * intersection < SPLIT_POLICY["bigram_jaccard_percent"] * union
            ):
                continue
            match, similarity = "near_duplicate", round(intersection / union, 4)

        matches.append(
            {
                "left_query_id": left,
                "right_query_id": right,
                "match": match,
                "similarity": similarity,
            }
        )
    return matches


def assign_splits(
    training_ids: set[str], test_ids: set[str], matches: list[ClaimMatch]
) -> dict[str, str]:
    parents = {query_id: query_id for query_id in training_ids}

    def root(query_id: str) -> str:
        while parents[query_id] != query_id:
            parents[query_id] = parents[parents[query_id]]
            query_id = parents[query_id]
        return query_id

    for pair in matches:
        left, right = pair["left_query_id"], pair["right_query_id"]
        if left in training_ids and right in training_ids:
            first, second = sorted((root(left), root(right)))
            parents[second] = first

    groups: dict[str, list[str]] = {}
    for query_id in sorted(training_ids):
        groups.setdefault(root(query_id), []).append(query_id)
    if len(groups) < 2:
        raise ValueError("at least two independent training query groups are required")

    seed = SPLIT_POLICY["seed"]
    ordered = sorted(
        groups,
        key=lambda query_id: (
            hashlib.sha256(f"{seed}:{query_id}".encode()).digest(),
            query_id,
        ),
    )
    totals = list(accumulate(len(groups[group]) for group in ordered[:-1]))
    target = len(training_ids) * SPLIT_POLICY["validation_percent"]
    boundary = min(
        range(len(totals)), key=lambda index: abs(100 * totals[index] - target)
    )
    validation_ids = {
        query_id for group in ordered[: boundary + 1] for query_id in groups[group]
    }

    return {
        **dict.fromkeys(training_ids - validation_ids, "train"),
        **dict.fromkeys(validation_ids, "validation"),
        **dict.fromkeys(test_ids, "test"),
    }
