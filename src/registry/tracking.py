"""Local MLflow tracking and registration for immutable SciFact bundles."""

from collections.abc import Iterator
from contextlib import contextmanager
from fcntl import LOCK_EX, flock
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mlflow import MlflowClient
from mlflow.entities.model_registry import RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.models import Model

from ..evaluation.gate import POLICY_VERSION, RECALL_TOLERANCE, GateDecision

TRACKING_ROOT = Path("artifacts/mlflow")
EXPERIMENT_NAME = "scifact-candidates"
MODEL_NAME = "scifact-dense-retriever"
MODEL_ARTIFACT = "model"
BUNDLE_ARTIFACT = f"{MODEL_ARTIFACT}/bundle"


@contextmanager
def registry_lock(root: Path) -> Iterator[None]:
    with (root / "registry.lock").open("a") as file:
        flock(file, LOCK_EX)
        yield


class LocalRegistry:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{self._root / 'tracking.db'}"
        with registry_lock(self._root):
            self.client = MlflowClient(tracking_uri=uri, registry_uri=uri)
            experiment = self.client.get_experiment_by_name(EXPERIMENT_NAME)
            self._experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else self.client.create_experiment(
                    EXPERIMENT_NAME,
                    artifact_location=(self._root / "artifacts").as_uri(),
                )
            )

    @property
    def tracking_root(self) -> Path:
        return self._root

    def start_run(self) -> str:
        return self.client.create_run(
            self._experiment_id, run_name="candidate evaluation"
        ).info.run_id

    def registered_model(self) -> RegisteredModel | None:
        try:
            return self.client.get_registered_model(MODEL_NAME)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            return None

    def champion_report(self, destination: Path) -> Path | None:
        model = self.registered_model()
        if model is None or "champion" not in model.aliases:
            return None
        version = self.client.get_model_version(MODEL_NAME, model.aliases["champion"])
        if not version.run_id:
            raise ValueError("the champion version has no tracking run")
        path = self.client.download_artifacts(
            version.run_id,
            f"{BUNDLE_ARTIFACT}/candidate-validation.json",
            str(destination),
        )
        return Path(path)

    def log_decision(
        self,
        run_id: str,
        reasons: tuple[str, ...],
        decision: GateDecision | None = None,
    ) -> None:
        status = "rejected" if reasons else "passed"
        self.client.set_tag(run_id, "gate.status", status)
        self.client.log_param(run_id, "gate.policy_version", POLICY_VERSION)
        self.client.log_param(run_id, "gate.recall_tolerance", RECALL_TOLERANCE)
        record: dict[str, Any] = {"status": status, "reasons": list(reasons)}
        if decision is not None:
            self.client.set_tag(run_id, "gate.reference", decision.reference)
            record.update(
                reference=decision.reference,
                floors=decision.floors,
                metrics=decision.metrics,
            )
            for method, metrics in decision.metrics.items():
                for metric, value in metrics.items():
                    self.client.log_metric(
                        run_id, f"{method}/{metric.replace('@', '_at_')}", value
                    )
        self.client.log_dict(run_id, record, "gate/decision.json")

    def log_candidate(self, run_id: str, path: Path, manifest: dict[str, Any]) -> None:
        self.client.set_tag(run_id, "candidate_id", manifest["candidate_id"])
        self.client.set_tag(run_id, "dataset_version", manifest["dataset_version"])
        self.client.log_param(
            run_id, "input_manifest_sha256", manifest["input_manifest_sha256"]
        )
        for name, value in manifest.get("training_config", {}).items():
            self.client.log_param(run_id, f"training.{name}", value)
        for name, value in manifest.get("training_stats", {}).items():
            self.client.log_metric(run_id, f"training/{name}", value)
        self.client.log_artifact(run_id, str(path), artifact_path="candidate_evidence")
        self.client.log_artifact(
            run_id,
            str(path.parent / "candidate-validation.json"),
            artifact_path="candidate_evidence",
        )

    def register(self, run_id: str, path: Path, manifest: dict[str, Any]) -> str:
        candidate_id = manifest["candidate_id"]
        with registry_lock(self._root):
            if self.registered_model() is None:
                self.client.create_registered_model(MODEL_NAME)
            versions = self.client.search_model_versions(
                f"name = '{MODEL_NAME}' AND tags.candidate_id = '{candidate_id}'",
                max_results=1,
            )
            if versions:
                return str(versions[0].version)

            self.client.log_artifacts(
                run_id, str(path.parent), artifact_path=BUNDLE_ARTIFACT
            )
            with TemporaryDirectory() as temporary:
                metadata = Path(temporary) / "MLmodel"
                model = Model(run_id=run_id, artifact_path=MODEL_ARTIFACT)
                model.add_flavor(
                    "scifact_dense",
                    manifest="bundle/manifest.json",
                    encoder="bundle/encoder",
                    index="bundle/index",
                )
                model.save(str(metadata))
                self.client.log_artifact(
                    run_id, str(metadata), artifact_path=MODEL_ARTIFACT
                )
            version = self.client.create_model_version(
                MODEL_NAME,
                source=f"runs:/{run_id}/{MODEL_ARTIFACT}",
                run_id=run_id,
                tags={
                    "status": "candidate",
                    "candidate_id": candidate_id,
                    "dataset_version": manifest["dataset_version"],
                },
            )
            return str(version.version)
