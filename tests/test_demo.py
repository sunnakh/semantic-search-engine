import argparse
import socket
import subprocess
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

DEMO_AVAILABLE = all(find_spec(name) for name in ("mlflow", "numpy", "fastapi"))
if DEMO_AVAILABLE:
    from src import demo
    from src.training.pipeline import PipelineResult


@unittest.skipUnless(DEMO_AVAILABLE, "dense, registry, and serving extras required")
class DemoTests(unittest.TestCase):
    def test_missing_monitoring_binary_stops_before_training(self) -> None:
        with patch("src.demo.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "install prometheus"):
                demo.check_prerequisites([8000, 9101, 9090, 3000], None)

    def test_duplicate_ports_are_rejected(self) -> None:
        with patch("src.demo.shutil.which", return_value="/unused"):
            with self.assertRaisesRegex(ValueError, "distinct ports"):
                demo.check_prerequisites([8000, 8000, 9090, 3000], None)

    def test_occupied_port_is_rejected_before_training(self) -> None:
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            with patch("src.demo.shutil.which", return_value="/unused"):
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    demo.check_prerequisites([port, 1, 2, 3], None)

    def test_failed_gate_never_exports_or_starts_services(self) -> None:
        with TemporaryDirectory() as directory:
            args = argparse.Namespace(
                api_port=8000,
                quality_port=9101,
                prometheus_port=9090,
                grafana_port=3000,
                grafana_home=None,
                output_root=Path(directory),
            )
            with (
                patch("src.demo.check_prerequisites", return_value=Path(directory)),
                patch("src.demo.ingest", return_value=Path("raw")),
                patch("src.demo.prepare", return_value=Path("prepared")),
                patch("src.demo.LocalRegistry"),
                patch(
                    "src.demo.run_pipeline",
                    return_value=PipelineResult(
                        "run", "candidate", None, ("regression",)
                    ),
                ),
                patch("src.demo.export_release") as export,
                patch("src.demo.subprocess.Popen") as start,
            ):
                with self.assertRaisesRegex(ValueError, "failed the gate"):
                    demo.run_demo(args)
                export.assert_not_called()
                start.assert_not_called()

    def test_dead_service_aborts_wait_and_cleanup_terminates_owned_process(
        self,
    ) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        try:
            demo.stop_process(process)
            self.assertIsNotNone(process.poll())
            with patch("src.demo.urlopen") as request:
                with self.assertRaisesRegex(RuntimeError, "service exited"):
                    demo.wait_for("http://127.0.0.1:1", [process], lambda body: True)
                request.assert_not_called()
        finally:
            demo.stop_process(process)
