import json
import unittest
from math import log, log1p, log2, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
from pyarrow import parquet

from src.data.integrity import sha256_file
from src.data.prepared import DOCUMENTS_FILE, VALIDATION_QRELS, VALIDATION_QUERIES
from src.documents import Document
from src.evaluation import benchmark as runner
from src.evaluation.metrics import score_ranking
from src.retrieval.keyword import KeywordHit, KeywordIndex
from src.retrieval.lexical import BM25Index, TfidfIndex


class KeywordTests(unittest.TestCase):
    def test_counts_distinct_tokens_and_breaks_ties_by_document_id(self) -> None:
        index = KeywordIndex(
            [
                Document("c", "", "rain rain rain"),
                Document("b", "rain", "soil"),
                Document("a", "SOIL", "RAIN rain"),
                Document("d", "", "rainfall soils"),
            ]
        )
        self.assertEqual(
            index.search("ＲＡＩＮ, soil rain"),
            [KeywordHit("a", 2), KeywordHit("b", 2), KeywordHit("c", 1)],
        )
        self.assertEqual(index.search("soil", top_k=1), [KeywordHit("a", 1)])
        self.assertEqual(index.search("???"), [])
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            index.search("soil", top_k=0)


class WeightedLexicalTests(unittest.TestCase):
    def test_tfidf_cosine_matches_hand_calculation(self) -> None:
        index = TfidfIndex(
            [
                Document("a", "alpha", "alpha beta"),
                Document("b", "alpha", "gamma"),
                Document("c", "", "beta"),
            ]
        )
        hits = index.search("alpha beta beta unknown")
        common = 1 + log(4 / 3)
        rare = 1 + log(4 / 2)
        self.assertEqual([hit.document_id for hit in hits], ["c", "a", "b"])
        self.assertAlmostEqual(hits[0].score, 2 / sqrt(5))
        self.assertAlmostEqual(hits[1].score, 4 / 5)
        self.assertAlmostEqual(hits[2].score, common / sqrt(5 * (common**2 + rare**2)))

    def test_bm25_scores_term_frequency_and_length(self) -> None:
        index = BM25Index(
            [
                Document("a", "rain", "clay"),
                Document("b", "rain rain", "sand rock clay gravel"),
                Document("c", "", "snow soil"),
            ]
        )
        hits = index.search("rain rain unknown")
        idf = log1p(1.5 / 2.5)
        self.assertEqual([hit.document_id for hit in hits], ["a", "b"])
        self.assertAlmostEqual(hits[0].score, idf * 2.2 / 1.84)
        self.assertAlmostEqual(hits[1].score, idf * 4.4 / 3.92)
        self.assertEqual(hits, index.search("rain"))

    def test_weighted_retrievers_omit_zero_scores_and_break_ties(self) -> None:
        documents = [Document("b", "rain", ""), Document("a", "", "rain")]
        for index_type in (TfidfIndex, BM25Index):
            with self.subTest(index=index_type.__name__):
                index = index_type(documents)
                self.assertEqual(
                    [hit.document_id for hit in index.search("rain", top_k=1)], ["a"]
                )
                self.assertEqual(index.search("unknown"), [])
                self.assertEqual(index_type([]).search("rain"), [])
                with self.assertRaisesRegex(ValueError, "top_k must be positive"):
                    index.search("rain", top_k=0)


class RankingMetricTests(unittest.TestCase):
    def test_scores_multiple_relevant_documents_with_cutoffs(self) -> None:
        metrics = score_ranking(
            ["x", "a", "y", "z", "w", "b", "d", "e", "f", "g", "c"], {"a", "b", "c"}
        )
        self.assertAlmostEqual(
            metrics["ndcg@10"], (1 / log2(3) + 1 / log2(7)) / (1 + 1 / log2(3) + 0.5)
        )
        self.assertEqual(metrics["recall@5"], 1 / 3)
        self.assertEqual(metrics["recall@10"], 2 / 3)
        self.assertEqual(metrics["mrr@10"], 0.5)

    def test_empty_ranking_scores_zero_and_duplicates_are_rejected(self) -> None:
        self.assertEqual(set(score_ranking([], {"a"}).values()), {0.0})
        with self.assertRaisesRegex(ValueError, "must be unique"):
            score_ranking(["a", "a"], {"a"})


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        hashes = {}
        for relative, rows in (
            (
                DOCUMENTS_FILE,
                [
                    {"document_id": "d1", "title": "Rain", "text": "Soil"},
                    {"document_id": "d2", "title": "River", "text": "Water"},
                ],
            ),
            (
                VALIDATION_QUERIES,
                [
                    {"query_id": "q1", "text": "RAIN"},
                    {"query_id": "q2", "text": "unmatched"},
                ],
            ),
            (
                VALIDATION_QRELS,
                [
                    {"query_id": "q1", "document_id": "d1", "relevance": 1},
                    {"query_id": "q2", "document_id": "d2", "relevance": 1},
                ],
            ),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(pa.Table.from_pylist(rows), path)
            hashes[relative] = sha256_file(path)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "dataset_version": "a" * 64,
                    "artifact_sha256": hashes,
                }
            ),
            encoding="utf-8",
        )

    def test_benchmarks_all_validation_queries_and_writes_identical_report(
        self,
    ) -> None:
        report = runner.benchmark(self.manifest)
        self.assertEqual(report["split"], "validation")
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["queries_with_relevant_hit_at_10"], 1)
        self.assertEqual(set(report["metrics"].values()), {0.5})
        self.assertEqual(report["per_query"][1]["ranked_document_ids"], [])
        output = self.root / "results" / "validation.json"
        runner.write_report(report, output)
        first = output.read_bytes()
        runner.write_report(runner.benchmark(self.manifest), output)
        self.assertEqual(output.read_bytes(), first)

    def test_rejects_changed_validation_labels(self) -> None:
        (self.root / VALIDATION_QRELS).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            runner.benchmark(self.manifest)

    def test_weighted_benchmarks_use_the_same_validation_inputs(self) -> None:
        keyword = runner.benchmark(self.manifest)
        for method in ("tfidf", "bm25"):
            with self.subTest(method=method):
                report = runner.benchmark(self.manifest, method)
                self.assertEqual(report["method"], method)
                self.assertEqual(
                    report["input_manifest_sha256"], keyword["input_manifest_sha256"]
                )
                self.assertEqual(report["per_query"], keyword["per_query"])
                if method == "bm25":
                    self.assertEqual(report["parameters"], {"k1": 1.2, "b": 0.75})

    def test_dense_rankings_use_the_shared_validation_evaluator(self) -> None:
        model = {"id": "fixture", "dimensions": 2}
        with patch.object(
            runner, "dense_rankings", return_value=([["d1", "d2"], ["d1", "d2"]], model)
        ) as rank:
            report = runner.benchmark(self.manifest, "dense")
        self.assertEqual(rank.call_args.args[1], ["RAIN", "unmatched"])
        self.assertEqual(report["method"], "dense")
        self.assertEqual(report["model"], model)
        self.assertEqual(report["queries_with_relevant_hit_at_10"], 2)
        self.assertEqual(report["metrics"]["mrr@10"], 0.75)
