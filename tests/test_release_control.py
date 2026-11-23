import json
import tempfile
import unittest
from pathlib import Path

from src.serving.control import ReleasePointer, ReleaseState, ReleaseStore


class ReleaseStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "release.json"
        self.store = ReleaseStore(self.path)
        self.first = ReleasePointer(1, 300, "a" * 64)
        self.second = ReleasePointer(2, 500, "b" * 64)

    def test_stage_promote_and_rollback_preserve_history(self) -> None:
        self.assertEqual(self.store.initialize(self.first).generation, 1)
        staged = self.store.stage(self.second, expected_generation=1)
        self.assertEqual(staged.shadow, self.second)
        promoted = self.store.promote(expected_generation=2)
        self.assertEqual(promoted.active, self.second)
        self.assertEqual(promoted.previous, self.first)
        self.assertIsNone(promoted.shadow)
        rolled_back = self.store.rollback(expected_generation=3)
        self.assertEqual(rolled_back.active, self.first)
        self.assertEqual(rolled_back.previous, self.second)
        self.assertEqual(self.store.read(), rolled_back)

    def test_stale_writer_and_invalid_state_are_rejected(self) -> None:
        self.store.initialize(self.first)
        with self.assertRaisesRegex(ValueError, "stale release state"):
            self.store.stage(self.second, expected_generation=2)
        self.path.write_text(json.dumps({"schema_version": 99}))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.store.read()

    def test_duplicate_release_roles_are_rejected(self) -> None:
        value = {
            "schema_version": 1,
            "generation": 2,
            "active": {
                "model_version": 1,
                "dense_timeout_ms": 300,
                "release_id": "a" * 64,
            },
            "previous": None,
            "shadow": {
                "model_version": 1,
                "dense_timeout_ms": 300,
                "release_id": "a" * 64,
            },
        }
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            ReleaseState.parse(value)


if __name__ == "__main__":
    unittest.main()
