"""Train or reuse a candidate, apply the quality gate, and record the decision."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from mlflow.exceptions import MlflowException

from ..data.prepared import DEFAULT_PROCESSED_ROOT, load_validation
from ..evaluation.gate import check_bundle, evaluate_candidate
from ..registry.tracking import TRACKING_ROOT, LocalRegistry
from ..serving.release import load_release
from .candidate import OUTPUT_ROOT, train_candidate
from .champion import served_release


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    candidate_id: str | None
    registered_version: str | None
    reasons: tuple[str, ...] = ()


def run_pipeline(
    prepared_manifest: Path,
    registry: LocalRegistry,
    candidate_manifest: Path | None = None,
    *,
    champion_api_url: str | None = None,
) -> PipelineResult:
    run_id = registry.start_run()
    try:
        path = candidate_manifest or train_candidate(prepared_manifest, OUTPUT_ROOT)
        try:
            data = load_validation(prepared_manifest)
            manifest = check_bundle(path, data)
            with TemporaryDirectory() as temporary:
                active = None
                champion: Path | None
                if champion_api_url is not None:
                    active = served_release(champion_api_url)
                    champion_release = load_release(
                        prepared_manifest,
                        registry.tracking_root,
                        int(active.model_version),
                        active.dense_timeout_ms,
                        Path(temporary),
                    )
                    if champion_release.metadata != active:
                        raise ValueError(
                            "serving and registry release lineage disagree"
                        )
                    champion = champion_release.bundle / "candidate-validation.json"
                    registry.client.set_tag(
                        run_id, "gate.champion_release_id", active.release_id
                    )
                    registry.client.set_tag(
                        run_id, "gate.champion_model_version", active.model_version
                    )
                    registry.client.set_tag(
                        run_id, "gate.champion_run_id", champion_release.registry_run_id
                    )
                else:
                    champion = registry.champion_report(Path(temporary))
                decision = evaluate_candidate(path, data, manifest, champion)
                if (
                    champion_api_url is not None
                    and active is not None
                    and served_release(champion_api_url) != active
                ):
                    raise ValueError(
                        "active serving release changed during evaluation; retry the gate"
                    )
        except (OSError, TypeError, ValueError) as error:
            reasons = (str(error),)
            registry.log_decision(run_id, reasons)
            result = PipelineResult(run_id, None, None, reasons)
        else:
            registry.log_candidate(run_id, path, manifest)
            registry.log_decision(run_id, decision.reasons, decision)
            version = None
            if decision.passed:
                version = registry.register(run_id, path, manifest)
                registry.client.set_tag(run_id, "registry.version", version)
            result = PipelineResult(
                run_id, manifest["candidate_id"], version, decision.reasons
            )
    except Exception as error:
        registry.client.set_tag(run_id, "pipeline.error_type", type(error).__name__)
        registry.client.set_terminated(run_id, "FAILED")
        raise

    registry.client.set_terminated(run_id, "FINISHED")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="prepared SciFact manifest")
    parser.add_argument(
        "--candidate", type=Path, help="evaluate an existing candidate manifest"
    )
    parser.add_argument("--tracking-root", type=Path, default=TRACKING_ROOT)
    parser.add_argument(
        "--champion-api-url",
        help="running API whose active model is the gate reference",
    )
    args = parser.parse_args()
    try:
        prepared = args.manifest
        if prepared is None:
            manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
            if len(manifests) != 1:
                raise ValueError(
                    "run make prepare or select a prepared snapshot with --manifest"
                )
            (prepared,) = manifests

        result = run_pipeline(
            prepared,
            LocalRegistry(args.tracking_root),
            args.candidate,
            champion_api_url=args.champion_api_url,
        )
    except (MlflowException, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Candidate pipeline failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), sort_keys=True))
    return 2 if result.reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
