"""Run the SciFact data, training, serving, and monitoring chain locally."""

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic, sleep
from urllib.parse import urlencode
from urllib.request import urlopen

from .data.ingest import DEFAULT_OUTPUT_ROOT, ingest
from .data.prepare import prepare
from .data.prepared import DEFAULT_PROCESSED_ROOT
from .registry.tracking import LocalRegistry
from .serving.client import search
from .serving.portable import export_release, load_portable_release
from .serving.schemas import SearchRequest
from .training.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONITORING_ROOT = PROJECT_ROOT / "infra/monitoring"


def check_prerequisites(ports: list[int], grafana_home: Path | None) -> Path:
    for program in ("prometheus", "grafana"):
        if shutil.which(program) is None:
            raise ValueError(f"install {program} before running make demo")
    if len(set(ports)) != len(ports):
        raise ValueError("demo services require four distinct ports")
    for port in ports:
        if not 1 <= port <= 65535:
            raise ValueError("ports must be between 1 and 65535")
        try:
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(
                f"demo port {port} is unavailable; select another port"
            ) from error
    if grafana_home is None:
        executable = shutil.which("grafana")
        assert executable is not None
        grafana_home = Path(executable).resolve().parents[1] / "share/grafana"
    if not (grafana_home / "conf/defaults.ini").is_file():
        raise ValueError("select the Grafana installation with --grafana-home")
    return grafana_home.resolve()


def stop_process(process: subprocess.Popen[bytes]) -> None:
    # Each service owns a process group, including the API's dense worker.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_for(
    url: str,
    processes: list[subprocess.Popen[bytes]],
    ready: Callable[[bytes], bool],
    timeout: int = 180,
) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("a demo service exited; inspect the run's log files")
        try:
            with urlopen(url, timeout=3) as response:
                value = response.read()
            if ready(value):
                return
        except (OSError, ValueError):
            pass
        sleep(0.5)
    raise TimeoutError(f"timed out waiting for {url}; inspect the run's log files")


def run_demo(args: argparse.Namespace) -> Path:
    grafana_home = check_prerequisites(
        [args.api_port, args.quality_port, args.prometheus_port, args.grafana_port],
        args.grafana_home,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_root = Path(mkdtemp(prefix="run-", dir=args.output_root)).resolve()
    print(f"Demo artifacts and logs: {run_root}", flush=True)

    print("Ingesting and validating pinned SciFact inputs", flush=True)
    prepared = prepare(ingest(DEFAULT_OUTPUT_ROOT), DEFAULT_PROCESSED_ROOT)
    print("Training, evaluating, and registering MiniLM", flush=True)
    tracking = run_root / "tracking"
    result = run_pipeline(prepared, LocalRegistry(tracking))
    if result.reasons or result.registered_version is None:
        raise ValueError(f"demo candidate failed the gate: {result.reasons}")
    bundle = export_release(
        prepared, tracking, int(result.registered_version), run_root / "release"
    )
    release = load_portable_release(bundle)
    api_url = f"http://127.0.0.1:{args.api_port}"
    prometheus_url = f"http://127.0.0.1:{args.prometheus_port}"
    grafana_url = f"http://127.0.0.1:{args.grafana_port}"
    config = run_root / "prometheus.yml"
    config.write_text(
        "global:\n  scrape_interval: 5s\n  evaluation_interval: 5s\n"
        f"rule_files:\n  - {json.dumps(str(MONITORING_ROOT / 'alerts.yml'))}\n"
        "scrape_configs:\n  - job_name: scifact-search\n    static_configs:\n"
        f"      - targets: ['127.0.0.1:{args.api_port}']\n"
        "  - job_name: scifact-sentinel\n    static_configs:\n"
        f"      - targets: ['127.0.0.1:{args.quality_port}']\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "SCIFACT_SENTINEL_TOKEN": secrets.token_hex(32),
        "GF_SECURITY_ADMIN_PASSWORD": secrets.token_hex(32),
        "GF_AUTH_ANONYMOUS_ENABLED": "true",
        "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
        "GF_PLUGINS_PREINSTALL_DISABLED": "true",
        "GF_SERVER_HTTP_ADDR": "127.0.0.1",
        "GF_SERVER_HTTP_PORT": str(args.grafana_port),
        "GF_PATHS_DATA": str(run_root / "grafana/data"),
        "GF_PATHS_LOGS": str(run_root / "grafana/log"),
        "GF_PATHS_PLUGINS": str(run_root / "grafana/plugins"),
        "GF_PATHS_PROVISIONING": str(MONITORING_ROOT / "grafana/provisioning"),
        "SCIFACT_GRAFANA_DASHBOARDS": str(MONITORING_ROOT / "grafana/dashboards"),
        "SCIFACT_PROMETHEUS_URL": prometheus_url,
    }
    processes: list[subprocess.Popen[bytes]] = []
    with ExitStack() as stack:

        def start(name: str, command: list[str]) -> None:
            log = stack.enter_context((run_root / f"{name}.log").open("wb"))
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            stack.callback(stop_process, process)
            processes.append(process)

        start(
            "api",
            [
                sys.executable,
                "-m",
                "src.serving.app",
                "--portable-release",
                str(bundle),
                "--port",
                str(args.api_port),
                "--monitor-drift",
            ],
        )
        wait_for(
            api_url + "/health/ready",
            processes,
            lambda body: (
                json.loads(body).get("release_id") == release.metadata.release_id
            ),
        )
        response = search(
            api_url, SearchRequest(query="exercise and heart health", top_k=5)
        )
        if response.fallback or response.method != "dense" or not response.results:
            raise ValueError("demo dense search failed its startup check")
        start(
            "quality",
            [
                sys.executable,
                "-m",
                "src.monitoring.sentinel",
                "--manifest",
                str(bundle / "snapshot/manifest.json"),
                "--reference-report",
                str(bundle / "bundle/candidate-validation.json"),
                "--api-url",
                api_url,
                "--port",
                str(args.quality_port),
            ],
        )
        wait_for(
            f"http://127.0.0.1:{args.quality_port}/metrics",
            processes,
            lambda body: b"scifact_sentinel_run_ok 1.0" in body,
        )
        start(
            "prometheus",
            [
                "prometheus",
                f"--config.file={config}",
                f"--web.listen-address=127.0.0.1:{args.prometheus_port}",
                f"--storage.tsdb.path={run_root / 'prometheus'}",
            ],
        )
        query = urlencode({"query": 'up{job=~"scifact-search|scifact-sentinel"}'})

        def targets_up(body: bytes) -> bool:
            value = json.loads(body)
            rows = value.get("data", {}).get("result", [])
            return len(rows) == 2 and all(row["value"][1] == "1" for row in rows)

        wait_for(prometheus_url + "/api/v1/query?" + query, processes, targets_up)
        start("grafana", ["grafana", "server", "--homepath", str(grafana_home)])
        wait_for(
            grafana_url + "/api/health",
            processes,
            lambda body: json.loads(body).get("database") == "ok",
        )
        wait_for(
            grafana_url + "/api/dashboards/uid/scifact-service-health",
            processes,
            lambda body: (
                json.loads(body).get("dashboard", {}).get("uid")
                == "scifact-service-health"
            ),
        )
        print(
            f"Demo ready. API: {api_url}/docs\nDashboard: {grafana_url}/d/scifact-service-health\n"
            f"Alerts: {prometheus_url}/alerts\nRelease: {release.metadata.release_id}",
            flush=True,
        )
        if args.check:
            return run_root
        print(
            "Press Ctrl+C to stop all demo services. Artifacts and logs are retained.",
            flush=True,
        )
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("a demo service exited; inspect the run's log files")
            sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/models/demo")
    )
    parser.add_argument("--grafana-home", type=Path)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--quality-port", type=int, default=9101)
    parser.add_argument("--prometheus-port", type=int, default=9090)
    parser.add_argument("--grafana-port", type=int, default=3000)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit after startup checks and stop all services",
    )
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run_demo(args)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
