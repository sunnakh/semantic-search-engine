import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

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
        hits = self.index.search("query", top_k=3)
        self.assertEqual([hit.document_id for hit in hits], ["b", "c", "a"])
        self.assertAlmostEqual(hits[0].score, 1)
        self.assertAlmostEqual(hits[2].score, 2**-0.5)

    def test_rejects_invalid_query_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero or non-finite"):
            self.index.rank_queries(["zero"])
        with self.assertRaisesRegex(ValueError, "same dimension"):
            self.index.rank_queries(["wrong dimension"])

    def test_saved_index_replays_rankings_and_rejects_altered_dimensions(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "index"
            self.index.save(directory)
            restored = DenseIndex.load(directory, FixtureEncoder())
            self.assertEqual(
                restored.rank_queries(["query"]), self.index.rank_queries(["query"])
            )
            np.save(
                directory / "vectors.npy",
                np.ones((1, 3), dtype=np.float32),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "one normalized float32 vector"):
                DenseIndex.load(directory, FixtureEncoder())
