"""Demonstrate a simulated quality incident, release rollback, and recovery."""

import argparse
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from urllib.parse import urlencode
from urllib.request import urlopen

from mlflow.exceptions import MlflowException

from ..data.prepared import DEFAULT_PROCESSED_ROOT
from ..demo import check_ports, start_service, wait_for, write_prometheus_config
from ..registry.tracking import TRACKING_ROOT
from ..serving.client import search
from ..serving.control import ReleasePointer, ReleaseStore
from ..serving.release import load_release
from ..serving.schemas import SearchRequest
from .fault_proxy import fault_proxy

ALERT = "SciFactSentinelQualityLow"
QUALITY_METRICS = {"ndcg_at_10", "recall_at_5"}


def alert_matches(body: bytes, firing: bool) -> bool:
    value = json.loads(body)
    alerts = [
        alert
        for alert in value["data"]["alerts"]
        if alert["labels"].get("alertname") == ALERT
    ]
    if not firing:
        return not alerts
    return {
        alert["labels"].get("metric") for alert in alerts if alert["state"] == "firing"
    } == QUALITY_METRICS


def quality_recovered(body: bytes) -> bool:
    value = json.loads(body)
    rows = value["data"]["result"]
    return {row["metric"].get("metric") for row in rows} == QUALITY_METRICS


def run_drill(args: argparse.Namespace) -> Path:
    if (
        args.good_version < 1
        or args.candidate_version < 1
        or args.good_version == args.candidate_version
    ):
        raise ValueError("select two distinct positive registered model versions")
    if shutil.which("prometheus") is None:
        raise ValueError("install prometheus before running the incident drill")
    check_ports(
        [args.api_port, args.proxy_port, args.quality_port, args.prometheus_port]
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="run-", dir=args.output_root)).resolve()
    steps: list[dict] = []
    evidence = {"simulated": True, "status": "running", "steps": steps}

    def record(name: str, details: dict) -> None:
        steps.append(
            {"phase": name, "at": datetime.now(timezone.utc).isoformat(), **details}
        )
        temporary = root / "evidence.tmp"
        temporary.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(root / "evidence.json")
        print(f"{name}: {root / 'evidence.json'}", flush=True)

    print(f"Simulated incident artifacts and logs: {root}", flush=True)
    try:
        good = load_release(
            args.manifest, args.tracking_root, args.good_version, 300, root / "good"
        )
        candidate = load_release(
            args.manifest,
            args.tracking_root,
            args.candidate_version,
            300,
            root / "candidate",
        )
        record(
            "verified_releases",
            {"good": asdict(good.metadata), "candidate": asdict(candidate.metadata)},
        )
        pointers = [
            ReleasePointer(
                int(release.metadata.model_version), 300, release.metadata.release_id
            )
            for release in (good, candidate)
        ]
        store = ReleaseStore(root / "state.json")
        store.initialize(pointers[0])
        api_url = f"http://127.0.0.1:{args.api_port}"
        prometheus_url = f"http://127.0.0.1:{args.prometheus_port}"
        config = write_prometheus_config(root, args.api_port, args.quality_port)
        environment = {**os.environ, "SCIFACT_SENTINEL_TOKEN": secrets.token_hex(32)}
        processes: list[subprocess.Popen[bytes]] = []

        def snapshot(url: str) -> dict:
            with urlopen(url, timeout=5) as response:
                return json.load(response)

        with ExitStack() as stack:

            def start_api() -> tuple[ExitStack, subprocess.Popen[bytes]]:
                owned = stack.enter_context(ExitStack())
                process = start_service(
                    owned,
                    root / "api.log",
                    [
                        sys.executable,
                        "-m",
                        "src.serving.app",
                        "--release-state",
                        str(store.path),
                        "--manifest",
                        str(args.manifest.resolve()),
                        "--tracking-root",
                        str(args.tracking_root.resolve()),
                        "--port",
                        str(args.api_port),
                        "--monitor-drift",
                    ],
                    environment,
                )
                processes.append(process)
                expected = store.read().active.release_id
                wait_for(
                    api_url + "/health/ready",
                    processes,
                    lambda body: json.loads(body).get("release_id") == expected,
                )
                return owned, process

            api_stack, api = start_api()
            proxy_url = stack.enter_context(
                fault_proxy(api_url, candidate.metadata.release_id, args.proxy_port)
            )
            processes.append(
                start_service(
                    stack,
                    root / "quality.log",
                    [
                        sys.executable,
                        "-m",
                        "src.monitoring.sentinel",
                        "--manifest",
                        str(args.manifest.resolve()),
                        "--reference-report",
                        str(good.bundle / "candidate-validation.json"),
                        "--api-url",
                        proxy_url,
                        "--port",
                        str(args.quality_port),
                        "--interval-seconds",
                        "3",
                    ],
                    environment,
                )
            )
            processes.append(
                start_service(
                    stack,
                    root / "prometheus.log",
                    [
                        "prometheus",
                        f"--config.file={config}",
                        f"--web.listen-address=127.0.0.1:{args.prometheus_port}",
                        f"--storage.tsdb.path={root / 'prometheus'}",
                    ],
                    environment,
                )
            )
            recovered_url = (
                prometheus_url
                + "/api/v1/query?"
                + urlencode(
                    {"query": "scifact_sentinel_score >= scifact_sentinel_floor"}
                )
            )
            alerts_url = prometheus_url + "/api/v1/alerts"
            wait_for(recovered_url, processes, quality_recovered)
            wait_for(alerts_url, processes, lambda body: alert_matches(body, False))
            record(
                "baseline_healthy",
                {
                    "release": snapshot(api_url + "/api/v1/models"),
                    "quality": snapshot(recovered_url),
                },
            )

            store.stage(pointers[1], expected_generation=1)
            store.promote(expected_generation=2)
            api_stack.close()
            processes.remove(api)
            api_stack, api = start_api()
            payload = SearchRequest(query="exercise and heart health", top_k=5)
            direct, injected = search(api_url, payload), search(proxy_url, payload)
            if direct.fallback or not direct.results or injected.results:
                raise ValueError(
                    "fault injection did not produce an isolated empty ranking"
                )
            record(
                "simulated_regression",
                {
                    "state": asdict(store.read()),
                    "worker_healthy": True,
                    "injected_results": len(injected.results),
                },
            )
            wait_for(alerts_url, processes, lambda body: alert_matches(body, True))
            record("quality_alerts_firing", {"alerts": snapshot(alerts_url)})

            # Follow the runbook by restoring the previous pointer and restarting.
            store.rollback(expected_generation=3)
            api_stack.close()
            processes.remove(api)
            api_stack, api = start_api()
            restored = snapshot(api_url + "/api/v1/models")
            if restored["release_id"] != good.metadata.release_id:
                raise ValueError("rollback did not restore the known-good release")
            response = search(proxy_url, payload)
            if (
                response.fallback
                or not response.results
                or response.release_id != good.metadata.release_id
            ):
                raise ValueError(
                    "rolled-back dense search failed while the fault remained armed"
                )
            wait_for(recovered_url, processes, quality_recovered)
            wait_for(alerts_url, processes, lambda body: alert_matches(body, False))
            record(
                "rollback_recovered",
                {
                    "release": restored,
                    "state": asdict(store.read()),
                    "quality": snapshot(recovered_url),
                    "alerts": snapshot(alerts_url),
                    "fault_still_armed_for": candidate.metadata.release_id,
                },
            )
        evidence["status"] = "passed"
        record("services_stopped", {})
    except BaseException as error:
        evidence["status"] = "failed"
        record("failed", {"error": str(error), "error_type": type(error).__name__})
        raise
    return root / "evidence.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tracking-root", type=Path, default=TRACKING_ROOT)
    parser.add_argument("--good-version", type=int, required=True)
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/models/incidents")
    )
    parser.add_argument("--api-port", type=int, default=18100)
    parser.add_argument("--proxy-port", type=int, default=18101)
    parser.add_argument("--quality-port", type=int, default=19101)
    parser.add_argument("--prometheus-port", type=int, default=19090)
    args = parser.parse_args()
    if args.manifest is None:
        manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
        if len(manifests) != 1:
            parser.error("prepare SciFact or select one snapshot with --manifest")
        (args.manifest,) = manifests

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        print(run_drill(args))
    except KeyboardInterrupt:
        return 130
    except (MlflowException, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Incident drill failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
