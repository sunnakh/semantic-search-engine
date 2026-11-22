import unittest
from importlib.util import find_spec

from src.documents import Document

DENSE_AVAILABLE = (
    find_spec("numpy") is not None and find_spec("sentence_transformers") is not None
)
if DENSE_AVAILABLE:
    import numpy as np
    from numpy.typing import NDArray

    from src.retrieval.dense import DenseIndex


class FixtureEncoder:
    def encode(self, texts: list[str]) -> "NDArray[np.float32]":
        vectors = {
            "A text": (1, 1),
            "B text": (5, 0),
            "C text": (10, 0),
            "D text": (-2, 0),
            "query": (1, 0),
            "zero": (0, 0),
            "wrong dimension": (1, 0, 0),
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


@unittest.skipUnless(DENSE_AVAILABLE, "dense extra required")
class DenseIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = DenseIndex(
            [
                Document("c", "C", "text"),
                Document("d", "D", "text"),
                Document("a", "A", "text"),
                Document("b", "B", "text"),
            ],
            FixtureEncoder(),
        )

    def test_exact_cosine_normalizes_and_resolves_ties(self) -> None:
        self.assertEqual(self.index.dimensions, 2)
        self.assertEqual(self.index.rank_queries(["query"]), [["b", "c", "a", "d"]])
        self.assertEqual(
            self.index.rank_queries(["query", "query"], top_k=2),
            [["b", "c"], ["b", "c"]],
        )
        self.assertEqual(self.index.rank_queries([]), [])

    def test_rejects_invalid_query_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero or non-finite"):
            self.index.rank_queries(["zero"])
        with self.assertRaisesRegex(ValueError, "same dimension"):
            self.index.rank_queries(["wrong dimension"])
