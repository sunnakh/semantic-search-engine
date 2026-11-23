"""Versioned reference and rolling-window inputs for query-length drift."""

from bisect import bisect_left

from prometheus_client import CollectorRegistry, Counter, Gauge

from ..data.prepared import PreparedSplit
from ..serving.schemas import MAX_QUERY_LENGTH

UPPER_BOUNDS = (32, 64, 128, 256, MAX_QUERY_LENGTH)
BUCKETS = tuple(
    f"{lower + 1}-{upper}"
    for lower, upper in zip((0, *UPPER_BOUNDS[:-1]), UPPER_BOUNDS, strict=True)
)


def reference_counts(training: PreparedSplit) -> list[int]:
    counts = [0] * len(BUCKETS)
    excluded = set(training.exact_test_overlap_ids)
    for query_id, query in training.queries.items():
        length = len(query.strip())
        if query_id not in excluded and 1 <= length <= MAX_QUERY_LENGTH:
            counts[bisect_left(UPPER_BOUNDS, length)] += 1
    if not sum(counts):
        raise ValueError("no eligible training claims for the drift reference")
    return counts


class QueryLengthMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.counter = Counter(
            "scifact_query_length_requests",
            "Completed non-sentinel search requests per disjoint query-length bucket",
            ["bucket", "release_id"],
            registry=registry,
        )
        self.reference = Gauge(
            "scifact_query_length_reference_fraction",
            "Share of eligible training claims per query-length bucket",
            ["bucket", "release_id"],
            registry=registry,
        )

    def configure(self, training: PreparedSplit, release_id: str) -> None:
        counts = reference_counts(training)
        total = sum(counts)
        self.counter.clear()
        self.reference.clear()
        self.requests = [
            self.counter.labels(bucket=bucket, release_id=release_id)
            for bucket in BUCKETS
        ]
        for bucket, count in zip(BUCKETS, counts, strict=True):
            self.reference.labels(bucket=bucket, release_id=release_id).set(
                count / total
            )

    def observe(self, length: int) -> None:
        self.requests[bisect_left(UPPER_BOUNDS, length)].inc()
