import json
import threading
import unittest
from contextlib import contextmanager
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
from pyarrow import parquet

from src.data.integrity import file_hashes, manifest_digest, sha256_file
from src.data.prepared import (
    DOCUMENTS_FILE,
    SPLIT_AUDIT,
    TEST_QRELS,
    TEST_QUERIES,
    VALIDATION_QRELS,
    VALIDATION_QUERIES,
    load_split,
    load_test,
    load_validation,
)
from src.evaluation.benchmark import evaluate_rankings, write_report

REGISTRY_AVAILABLE = all(
    find_spec(name) for name in ("mlflow", "numpy", "sentence_transformers")
)
if REGISTRY_AVAILABLE:
    import numpy as np

    from src.evaluation import final
    from src.evaluation.gate import (
        check_bundle,
        evaluate_ranking_gate,
        verified_metrics,
    )
    from src.registry.tracking import BUNDLE_ARTIFACT, MODEL_NAME, LocalRegistry
    from src.serving.control import ReleasePointer, ReleaseStore
    from src.serving.portable import export_release, load_portable_release
    from src.serving.release import load_release
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

    def bundle(
        self,
        *,
        worse: bool = False,
        name: str | None = None,
        commit: str = "a" * 40,
        second_place: bool = False,
        weaker_baseline: bool = False,
    ) -> Path:
        bundle = self.root / (name or ("worse" if worse else "candidate"))
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
        if second_place:
            vectors[[0, 1]] = vectors[[1, 0]]
        np.save(bundle / "index/vectors.npy", vectors, allow_pickle=False)
        ranked_ids = [self.ids[index] for index in np.argsort(-vectors[:, 0])[:10]]

        base: dict[str, str | int] = {"id": "fixture", "revision": "fixture-revision"}
        lineage = {
            "manifest_version": 1,
            "dataset_version": self.data.dataset_version,
            "input_manifest_sha256": self.data.manifest_sha256,
            "base_model": base,
            "inference": {"dimensions": 2},
            "source": {"commit": commit},
            "model_and_index_sha256": file_hashes(bundle),
        }
        candidate_id = manifest_digest(lineage)
        candidate = evaluate_rankings(self.data, [ranked_ids] * 2, "fine_tuned_minilm")
        candidate["model"] = {"candidate_id": candidate_id}
        baseline = self.ids[:10]
        if weaker_baseline:
            baseline[0], baseline[1] = baseline[1], baseline[0]
        bm25 = evaluate_rankings(self.data, [baseline] * 2, "bm25")
        frozen = evaluate_rankings(self.data, [baseline] * 2, "dense")
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

    @contextmanager
    def serving_api(self, payloads):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payloads[0]).encode())
                if len(payloads) > 1:
                    payloads.pop(0)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_gate_follows_served_release_after_promotion_and_rollback(self) -> None:
        candidate = self.bundle()
        first = run_pipeline(self.prepared, self.registry, candidate)
        second_candidate = self.bundle(name="second", commit="b" * 40)
        second = run_pipeline(self.prepared, self.registry, second_candidate)
        self.assertEqual(
            (first.registered_version, second.registered_version), ("1", "2")
        )
        with TemporaryDirectory() as temporary:
            releases = [
                load_release(
                    self.prepared,
                    self.root / "tracking",
                    version,
                    300,
                    Path(temporary) / str(version),
                )
                for version in (1, 2)
            ]
        pointers = [
            ReleasePointer(
                int(release.metadata.model_version), 300, release.metadata.release_id
            )
            for release in releases
        ]
        store = ReleaseStore(self.root / "release.json")
        store.initialize(pointers[0])
        # The legacy alias stays on version 2 throughout the drill.
        self.registry.client.set_registered_model_alias(MODEL_NAME, "champion", "2")
        for version in (1, 2, 1):
            if version == 2:
                store.stage(pointers[1], 1)
                store.promote(2)
            elif store.read().generation == 3:
                store.rollback(3)
            active = store.read().active
            metadata = asdict(releases[active.model_version - 1].metadata)
            # Model the metadata returned after the API reloads the selected release.
            with self.serving_api([metadata]) as url:
                result = run_pipeline(
                    self.prepared, self.registry, candidate, champion_api_url=url
                )
            run = self.registry.client.get_run(result.run_id)
            self.assertFalse(result.reasons)
            self.assertEqual(run.data.tags["gate.reference"], "champion")
            self.assertEqual(run.data.tags["gate.champion_model_version"], str(version))
            self.assertEqual(
                run.data.tags["gate.champion_release_id"], active.release_id
            )

    def test_worse_than_served_champion_is_blocked_even_if_baselines_pass(self) -> None:
        champion = self.bundle(weaker_baseline=True)
        first = run_pipeline(self.prepared, self.registry, champion)
        self.assertEqual(first.registered_version, "1")
        challenger = self.bundle(
            name="challenger", commit="b" * 40, second_place=True, weaker_baseline=True
        )
        baseline_result = run_pipeline(self.prepared, self.registry, challenger)
        self.assertFalse(baseline_result.reasons)
        self.registry.client.set_registered_model_alias(MODEL_NAME, "champion", "2")
        with TemporaryDirectory() as temporary:
            active = load_release(
                self.prepared, self.root / "tracking", 1, 300, Path(temporary)
            ).metadata
        with (
            self.serving_api([asdict(active)]) as url,
            patch.object(self.registry, "register") as register,
        ):
            result = run_pipeline(
                self.prepared, self.registry, challenger, champion_api_url=url
            )
        self.assertIsNone(result.registered_version)
        self.assertTrue(any("ndcg@10" in reason for reason in result.reasons))
        register.assert_not_called()

    def test_missing_changed_or_mismatched_served_champion_blocks_registration(
        self,
    ) -> None:
        candidate = self.bundle()
        first = run_pipeline(self.prepared, self.registry, candidate)
        self.assertEqual(first.registered_version, "1")
        with TemporaryDirectory() as temporary:
            active = load_release(
                self.prepared, self.root / "tracking", 1, 300, Path(temporary)
            ).metadata
        metadata = asdict(active)
        changed = {**metadata, "release_id": "f" * 64}
        for payloads in ([{}], [changed], [metadata, changed]):
            with self.subTest(payloads=payloads), self.serving_api(payloads) as url:
                result = run_pipeline(
                    self.prepared, self.registry, candidate, champion_api_url=url
                )
                self.assertIsNone(result.registered_version)
                self.assertTrue(result.reasons)
        with self.serving_api([metadata]) as stopped_url:
            pass
        result = run_pipeline(
            self.prepared, self.registry, candidate, champion_api_url=stopped_url
        )
        self.assertIsNone(result.registered_version)
        self.assertIn("cannot resolve", result.reasons[0])

    def test_ranking_gate_rejects_worse_report(self) -> None:
        bundle = self.bundle(worse=True).parent
        decision = evaluate_ranking_gate(
            bundle / "candidate-validation.json",
            self.data,
            "fine_tuned_minilm",
            bundle / "bm25-validation.json",
            "bm25",
            bundle / "frozen_minilm-validation.json",
            "dense",
        )
        self.assertFalse(decision.passed)
        self.assertEqual(len(decision.reasons), 2)
        self.assertEqual(decision.reference, "dense")

    def test_official_test_is_kept_out_of_candidate_workflows(self) -> None:
        path = self.bundle()
        with self.assertRaisesRegex(ValueError, "only train and validation"):
            load_split(self.prepared, "test")
        data = replace(self.data, split="test")
        with self.assertRaisesRegex(ValueError, "validation inputs"):
            check_bundle(path, data)
        with self.assertRaisesRegex(ValueError, "validation inputs"):
            verified_metrics(
                path.parent / "candidate-validation.json", data, "fine_tuned_minilm"
            )

    def test_final_benchmark_replays_pinned_model_without_changing_registry(
        self,
    ) -> None:
        snapshot = self.prepared.parent
        for name, rows in (
            (
                TEST_QUERIES,
                [{"query_id": qid, "text": "test claim"} for qid in ("t1", "t2")],
            ),
            (
                TEST_QRELS,
                [
                    {"query_id": qid, "document_id": self.ids[index], "relevance": 1}
                    for index, qid in enumerate(("t1", "t2"))
                ],
            ),
        ):
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_table(pa.Table.from_pylist(rows), target)
        (snapshot / SPLIT_AUDIT).write_text(
            json.dumps(
                {
                    "official_test_matches": [
                        {
                            "left_query_id": "q1",
                            "right_query_id": "t1",
                            "match": "exact_normalized",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.prepared.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "dataset_version": "b" * 64,
                    "artifact_sha256": {
                        name: digest
                        for name, digest in file_hashes(snapshot).items()
                        if name != "manifest.json"
                    },
                }
            ),
            encoding="utf-8",
        )
        self.data = load_validation(self.prepared)
        candidate = self.bundle()
        result = run_pipeline(self.prepared, self.registry, candidate)
        self.assertEqual(result.registered_version, "1")
        before = file_hashes(snapshot)
        candidate_before = file_hashes(candidate.parent)
        experiment_id = self.registry.client.get_run(result.run_id).info.experiment_id
        runs_before = self.registry.client.search_runs([experiment_id])
        output = self.root / "official-test"

        def open_test(manifest):
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["release"]["model_version"], "1")
            return load_test(manifest)

        with (
            patch.object(final, "load_test", side_effect=open_test),
            patch.object(
                final,
                "dense_rankings",
                return_value=([self.ids[:10]] * 2, {"id": "fixture"}),
            ),
            patch.object(final, "load_minilm"),
            patch.object(
                final.MiniLMEncoder,
                "encode",
                side_effect=lambda texts: np.tile(
                    np.array([[1, 0]], dtype=np.float32), (len(texts), 1)
                ),
            ),
        ):
            report_path = final.run_test_benchmark(
                self.prepared, self.root / "tracking", 1, output
            )
            original = report_path.read_bytes()
            final.run_test_benchmark(self.prepared, self.root / "tracking", 1, output)
            self.assertEqual(original, report_path.read_bytes())
            with patch.object(
                final, "source_provenance", return_value={"commit": "changed"}
            ):
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    final.run_test_benchmark(
                        self.prepared, self.root / "tracking", 1, output
                    )
        report = json.loads(original)
        chosen = report["reports"]["fine_tuned_minilm"]
        self.assertEqual(chosen["split"], "test")
        self.assertEqual(chosen["query_count"], 2)
        self.assertEqual(chosen["metrics"]["mrr@10"], 0.75)
        sensitivity = report["excluding_audited_overlap"]["fine_tuned_minilm"]
        self.assertEqual(sensitivity["query_count"], 1)
        self.assertEqual(sensitivity["metrics"]["mrr@10"], 0.5)
        self.assertEqual(before, file_hashes(snapshot))
        self.assertEqual(candidate_before, file_hashes(candidate.parent))
        self.assertEqual(
            len(runs_before),
            len(self.registry.client.search_runs([experiment_id])),
        )
        self.assertEqual(
            len(self.registry.client.search_model_versions(f"name = '{MODEL_NAME}'")), 1
        )
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            final.save_once({"changed": True}, report_path)
        (snapshot / SPLIT_AUDIT).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            final.overlap_summary(self.prepared, {"t1", "t2"})
        (snapshot / TEST_QRELS).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_test(self.prepared)

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

    def test_portable_release_can_move_and_rejects_corruption(self) -> None:
        tracking = self.root / "tracking"
        result = run_pipeline(self.prepared, self.registry, self.bundle())
        self.assertEqual(result.registered_version, "1")
        exported = export_release(self.prepared, tracking, 1, self.root / "export", 420)
        with self.assertRaises(FileExistsError):
            export_release(self.prepared, tracking, 1, exported)

        relocated = exported.rename(self.root / "mounted elsewhere")
        self.prepared.parent.rename(self.root / "original snapshot moved")
        with patch("src.serving.release.MlflowClient", side_effect=AssertionError):
            release = load_portable_release(relocated)
        self.assertEqual(release.bundle, relocated / "bundle")
        self.assertEqual(release.registry_run_id, result.run_id)
        self.assertEqual(release.metadata.model_version, "1")
        self.assertEqual(release.metadata.training_commit, "a" * 40)
        self.assertEqual(release.metadata.dataset_version, self.data.dataset_version)
        self.assertEqual(release.metadata.dense_timeout_ms, 420)
        self.assertEqual(sorted(release.documents), self.ids)

        manifest_path = relocated / "release.json"
        manifest = json.loads(manifest_path.read_text())
        original_manifest = manifest_path.read_bytes()
        manifest["release"]["training_commit"] = "b" * 40
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "lineage does not match"):
            load_portable_release(relocated)
        manifest_path.write_bytes(original_manifest)

        link = relocated / "linked-snapshot"
        link.symlink_to(relocated / "snapshot", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "regular artifact"):
            load_portable_release(relocated)
        link.unlink()

        index = release.bundle / "index/vectors.npy"
        index.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "portable release checksum mismatch"):
            load_portable_release(relocated)

    def test_portable_export_requires_a_passing_registry_run(self) -> None:
        result = run_pipeline(self.prepared, self.registry, self.bundle())
        self.assertEqual(result.registered_version, "1")
        self.registry.client.set_tag(result.run_id, "gate.status", "rejected")
        destination = self.root / "export"
        with self.assertRaisesRegex(ValueError, "passing gate run"):
            export_release(self.prepared, self.root / "tracking", 1, destination)
        self.assertFalse(destination.exists())
