import hashlib
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.data import ingest


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_root = Path(temporary.name)
        judgments = (
            b"query-id\tcorpus-id\tscore\n001\tdoc-a\t1\n001\tdoc-b\t1\n1\tdoc-a\t1\n"
        )
        digest = hashlib.sha256(judgments).hexdigest()
        self.sources = (
            ingest.SourceFile("example/qrels", "fixture", "train.tsv", digest),
            ingest.SourceFile("example/qrels", "fixture", "test.tsv", digest),
        )
        self.responses = {source.url: judgments for source in self.sources}

    def run_ingestion(self) -> Path:
        def open_fixture(url: str, *, timeout: int) -> BytesIO:
            return BytesIO(self.responses[url])

        with (
            patch.object(ingest, "SOURCES", self.sources),
            patch.object(ingest, "urlopen", side_effect=open_fixture),
        ):
            return ingest.ingest(self.output_root)

    def test_identical_rerun_reuses_snapshot_and_counts_query_ids_as_text(self) -> None:
        manifest_path = self.run_ingestion()
        original = manifest_path.read_bytes()

        self.assertEqual(self.run_ingestion(), manifest_path)
        self.assertEqual(manifest_path.read_bytes(), original)
        self.assertEqual(list(self.output_root.iterdir()), [manifest_path.parent])
        manifest = json.loads(original)
        self.assertEqual(manifest["dataset_version"], manifest_path.parent.name)
        self.assertEqual(manifest["sources"][0]["row_count"], 3)
        self.assertEqual(manifest["sources"][0]["distinct_judged_queries"], 2)

    def test_bad_download_leaves_no_snapshot(self) -> None:
        self.responses[self.sources[1].url] = b"corrupted download"

        with self.assertRaisesRegex(
            ValueError, "checksum mismatch for downloaded source"
        ):
            self.run_ingestion()

        self.assertEqual(list(self.output_root.iterdir()), [])

    def test_rerun_detects_corruption_in_published_snapshot(self) -> None:
        manifest_path = self.run_ingestion()
        published_file = manifest_path.parent / self.sources[0].path
        published_file.write_bytes(b"corrupted local file")

        with self.assertRaisesRegex(
            ValueError, "checksum mismatch in existing snapshot"
        ):
            self.run_ingestion()

        self.assertEqual(list(self.output_root.iterdir()), [manifest_path.parent])
