import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
from pyarrow import parquet

from src.data import ingest, prepare, splits
from src.data.integrity import manifest_digest, sha256_file
from src.data.prepared import DOCUMENTS_FILE, TRAIN_QRELS, TRAIN_QUERIES, load_split


class PrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.output = self.root / "processed"

    def raw_manifest(self, test: str = "held\td2\t1\n") -> Path:
        documents = {"d1": "Rain affects soil", "d2": "A report", "blank": " "}
        rows = [
            {"_id": key, "title": "", "text": text} for key, text in documents.items()
        ]
        rows.extend([{"_id": "dup", "title": "", "text": "Duplicate ID"}] * 2)
        queries = {
            "a": "Rain affects soil.",
            "b": "RAIN AFFECTS SOIL",
            "c": "Birds migrate",
            "d": "Ocean currents move",
            "held": "Rain affects soil",
            "unjudged": "No labels",
        }
        for name, records in (
            (ingest.CORPUS_PATH, rows),
            (
                ingest.QUERIES_PATH,
                [{"_id": key, "text": text} for key, text in queries.items()],
            ),
        ):
            path = self.raw / name
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(pa.Table.from_pylist(records), path)

        train = (
            "a\td1\t1\na\td2\t1\nb\td1\t1\nc\td2\t1\nd\td2\t1\n"
            "a\tmissing\t1\na\td1\t1\nb\td1\t0\nc\tblank\t1\na\tdup\t1\n"
        )
        header = "query-id\tcorpus-id\tscore\n"
        for split, judgments in (("train", train), ("test", test)):
            (self.raw / f"{split}.tsv").write_text(header + judgments, encoding="utf-8")

        self.sources = tuple(
            ingest.SourceFile(
                source.repository,
                source.revision,
                source.path,
                sha256_file(self.raw / source.path),
            )
            for source in ingest.SOURCES
        )
        inventory = {
            "dataset": "scifact",
            "manifest_version": 1,
            "sources": [
                {**vars(source), **ingest.count_rows(self.raw / source.path)}
                for source in self.sources
            ],
        }
        manifest = self.raw / "manifest.json"
        manifest.write_text(
            json.dumps({**inventory, "dataset_version": manifest_digest(inventory)}),
            encoding="utf-8",
        )
        return manifest

    def run_prepare(self, manifest: Path) -> Path:
        with patch.object(prepare, "SOURCES", self.sources):
            return prepare.prepare(manifest, self.output)

    def test_quarantine_and_deterministic_split_preserve_official_benchmark(
        self,
    ) -> None:
        raw = self.raw_manifest()
        raw_hashes = prepare.file_hashes(self.raw)
        manifest = self.run_prepare(raw)
        snapshot = manifest.parent
        artifact_hashes = prepare.file_hashes(snapshot)
        self.assertEqual(self.run_prepare(raw), manifest)
        self.assertEqual(prepare.file_hashes(snapshot), artifact_hashes)
        self.assertEqual(prepare.file_hashes(self.raw), raw_hashes)

        rejected = [
            json.loads(line)
            for line in (snapshot / "quarantine.jsonl").read_text().splitlines()
        ]
        self.assertCountEqual(
            [(row["row"], row["reason"]) for row in rejected],
            [
                (3, "blank_or_invalid_text"),
                (4, "duplicate_id"),
                (5, "duplicate_id"),
                (7, "unknown_document_id"),
                (8, "duplicate_qrel"),
                (9, "invalid_relevance"),
                (10, "unknown_document_id"),
                (11, "unknown_document_id"),
            ],
        )
        queries = parquet.read_table(snapshot / "queries").to_pylist()
        membership = {row["query_id"]: row["split"] for row in queries}
        self.assertEqual(set(membership), {"a", "b", "c", "d", "held"})
        self.assertEqual(membership["held"], "test")
        self.assertEqual(membership["a"], membership["b"])
        self.assertEqual(set(membership.values()), {"train", "validation", "test"})
        qrels = parquet.read_table(snapshot / "qrels").to_pylist()
        self.assertEqual(
            {(row["query_id"], row["document_id"]) for row in qrels},
            {
                ("a", "d1"),
                ("a", "d2"),
                ("b", "d1"),
                ("c", "d2"),
                ("d", "d2"),
                ("held", "d2"),
            },
        )
        self.assertTrue(
            all(row["split"] == membership[row["query_id"]] for row in qrels)
        )
        audit = json.loads((snapshot / "split_audit.json").read_text())
        self.assertEqual(audit["unjudged_query_ids"], ["unjudged"])
        self.assertEqual(len(audit["official_test_matches"]), 2)

        with patch(
            "src.data.prepared.parquet.ParquetFile", wraps=parquet.ParquetFile
        ) as reader:
            training = load_split(manifest, "train")
        self.assertEqual(
            [call.args[0] for call in reader.call_args_list],
            [snapshot / name for name in (DOCUMENTS_FILE, TRAIN_QUERIES, TRAIN_QRELS)],
        )
        self.assertEqual(
            set(training.queries),
            {qid for qid, split in membership.items() if split == "train"},
        )
        self.assertEqual(
            set(training.exact_test_overlap_ids), set(training.queries) & {"a", "b"}
        )

    def test_overlap_between_official_splits_blocks_publication(self) -> None:
        raw = self.raw_manifest("a\td1\t1\nheld\td2\t1\n")
        with self.assertRaisesRegex(ValueError, "both official splits"):
            self.run_prepare(raw)
        self.assertFalse(list(self.output.glob("*/manifest.json")))

    def test_invalid_test_qrel_blocks_publication_and_retains_quarantine(self) -> None:
        raw = self.raw_manifest("held\td2\t1\nheld\tmissing\t1\n")
        with self.assertRaisesRegex(ValueError, "change the official test benchmark"):
            self.run_prepare(raw)
        self.assertFalse(list(self.output.glob("*/manifest.json")))
        (report,) = self.output.glob("rejected/*/quarantine.jsonl")
        rejected = [json.loads(line) for line in report.read_text().splitlines()]
        self.assertEqual(
            [
                (row["row"], row["reason"])
                for row in rejected
                if row["source"] == "test.tsv"
            ],
            [(3, "unknown_document_id")],
        )

    def test_rejects_changed_raw_bytes_before_publication(self) -> None:
        raw = self.raw_manifest()
        (self.raw / "train.tsv").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "raw file checksum mismatch"):
            self.run_prepare(raw)
        self.assertFalse(self.output.exists())

    def test_near_duplicate_development_claims_stay_together(self) -> None:
        queries = {
            "a": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
            "b": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda xi",
            "c": "A separate claim",
        }
        matches = splits.audit_claims(queries, set())
        self.assertEqual(
            [
                (match["left_query_id"], match["right_query_id"], match["match"])
                for match in matches
            ],
            [("a", "b", "near_duplicate")],
        )
        membership = splits.assign_splits(set(queries), set(), matches)
        self.assertEqual(membership["a"], membership["b"])
        self.assertNotEqual(membership["a"], membership["c"])
