import json
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
from pyarrow import parquet

from src.data.integrity import file_hashes, manifest_digest, sha256_file
from src.data.prepared import (
    DOCUMENTS_FILE,
    VALIDATION_QRELS,
    VALIDATION_QUERIES,
    load_validation,
)
from src.evaluation.benchmark import evaluate_rankings, write_report

REGISTRY_AVAILABLE = all(
    find_spec(name) for name in ("mlflow", "numpy", "sentence_transformers")
)
if REGISTRY_AVAILABLE:
    import numpy as np

    from src.registry.tracking import BUNDLE_ARTIFACT, MODEL_NAME, LocalRegistry
    from src.training.pipeline import run_pipeline


@unittest.skipUnless(REGISTRY_AVAILABLE, "dense and registry extras required")
class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = LocalRegistry(self.root / "tracking")
        self.ids = [f"d{index:02d}" for index in range(11)]
        snapshot = self.root / "prepared"
        datasets = {
            DOCUMENTS_FILE: [
                {"document_id": document_id, "title": "", "text": document_id}
                for document_id in self.ids
            ],
            VALIDATION_QUERIES: [
                {"query_id": "q1", "text": "claim one"},
                {"query_id": "q2", "text": "claim two"},
            ],
            VALIDATION_QRELS: [
                {"query_id": qid, "document_id": self.ids[0], "relevance": 1}
                for qid in ("q1", "q2")
            ],
        }
        for location, rows in datasets.items():
            path = snapshot / location
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(pa.Table.from_pylist(rows), path)
        self.prepared = snapshot / "manifest.json"
        self.prepared.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "dataset_version": "b" * 64,
                    "artifact_sha256": file_hashes(snapshot),
                }
            ),
            encoding="utf-8",
        )
        self.data = load_validation(self.prepared)

    def bundle(self, *, worse: bool = False) -> Path:
        bundle = self.root / ("worse" if worse else "candidate")
        (bundle / "encoder/1_Pooling").mkdir(parents=True)
        (bundle / "index").mkdir()
        (bundle / "encoder/model.safetensors").write_bytes(b"test fixture checkpoint")
        (bundle / "encoder/modules.json").write_text("[]", encoding="utf-8")
        (bundle / "encoder/tokenizer.json").write_text("{}", encoding="utf-8")
        (bundle / "encoder/1_Pooling/config.json").write_text(
            '{"embedding_dimension": 2}', encoding="utf-8"
        )
        (bundle / "index/document_ids.json").write_text(
            json.dumps(self.ids), encoding="utf-8"
        )
        cosine = np.linspace(1, -1, len(self.ids), dtype=np.float32)
        if worse:
            cosine = -cosine
        vectors = np.column_stack((cosine, np.sqrt(1 - cosine**2)))
        np.save(bundle / "index/vectors.npy", vectors, allow_pickle=False)
        ranked_ids = [self.ids[index] for index in np.argsort(-vectors[:, 0])[:10]]

        base: dict[str, str | int] = {"id": "fixture", "revision": "fixture-revision"}
        lineage = {
            "manifest_version": 1,
            "dataset_version": self.data.dataset_version,
            "input_manifest_sha256": self.data.manifest_sha256,
            "base_model": base,
            "inference": {"dimensions": 2},
            "model_and_index_sha256": file_hashes(bundle),
        }
        candidate_id = manifest_digest(lineage)
        candidate = evaluate_rankings(self.data, [ranked_ids] * 2, "fine_tuned_minilm")
        candidate["model"] = {"candidate_id": candidate_id}
        bm25 = evaluate_rankings(self.data, [self.ids[:10]] * 2, "bm25")
        frozen = evaluate_rankings(self.data, [self.ids[:10]] * 2, "dense")
        frozen["model"] = base
        reports = {"candidate": candidate, "bm25": bm25, "frozen_minilm": frozen}
        for name, report in reports.items():
            write_report(report, bundle / f"{name}-validation.json")
        manifest = bundle / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    **lineage,
                    "candidate_id": candidate_id,
                    "status": "evaluated",
                    "validation_metrics": {
                        name: report["metrics"] for name, report in reports.items()
                    },
                    "artifact_sha256": file_hashes(bundle),
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_deliberately_worse_rankings_are_logged_and_never_registered(self) -> None:
        result = run_pipeline(self.prepared, self.registry, self.bundle(worse=True))
        run = self.registry.client.get_run(result.run_id)
        self.assertIsNone(result.registered_version)
        self.assertEqual(len(result.reasons), 2)
        self.assertEqual(run.data.tags["gate.status"], "rejected")
        self.assertEqual(run.data.metrics["candidate/ndcg_at_10"], 0)
        self.assertEqual(run.info.status, "FINISHED")
        self.assertIsNone(self.registry.registered_model())

    def test_passing_bundle_registers_once_and_can_be_the_reference(self) -> None:
        path = self.bundle()
        first = run_pipeline(self.prepared, self.registry, path)
        self.assertEqual(first.registered_version, "1")
        self.assertEqual(
            self.registry.client.get_run(first.run_id).data.tags["gate.reference"],
            "frozen_minilm",
        )
        stored = (
            Path(self.registry.client.download_artifacts(first.run_id, BUNDLE_ARTIFACT))
            / "manifest.json"
        )
        self.assertEqual(sha256_file(stored), sha256_file(path))
        self.assertEqual(file_hashes(stored.parent), file_hashes(path.parent))
        model = self.registry.registered_model()
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.aliases, {})

        self.registry.client.set_registered_model_alias(MODEL_NAME, "champion", "1")
        replay = run_pipeline(self.prepared, self.registry, path)
        self.assertNotEqual(replay.run_id, first.run_id)
        self.assertEqual(replay.registered_version, "1")
        self.assertEqual(
            self.registry.client.get_run(replay.run_id).data.tags["gate.reference"],
            "champion",
        )
        self.assertEqual(
            len(self.registry.client.search_model_versions(f"name = '{MODEL_NAME}'")), 1
        )

    def test_corrupted_artifact_is_rejected(self) -> None:
        path = self.bundle()
        (path.parent / "index/vectors.npy").write_bytes(b"corrupted")
        result = run_pipeline(self.prepared, self.registry, path)
        self.assertIn("checksum mismatch", result.reasons[0])
        self.assertEqual(
            self.registry.client.get_run(result.run_id).data.tags["gate.status"],
            "rejected",
        )
        self.assertIsNone(self.registry.registered_model())
