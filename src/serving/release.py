"""Resolve a pinned, gate-approved model and build its lexical fallback."""

import re
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient

from .. import text
from ..data.integrity import manifest_digest, sha256_file
from ..data.prepared import load_validation
from ..documents import Document
from ..evaluation.gate import check_bundle
from ..registry.tracking import BUNDLE_ARTIFACT, MODEL_NAME
from ..retrieval import keyword, lexical


@dataclass(frozen=True)
class Release:
    release_id: str
    model_version: str
    candidate_id: str
    training_commit: str
    dataset_version: str
    prepared_manifest_sha256: str
    bundle_manifest_sha256: str
    keyword_version: str
    bm25_version: str
    dense_timeout_ms: int


@dataclass(frozen=True)
class LoadedRelease:
    metadata: Release
    bundle: Path
    documents: dict[str, Document]
    keyword: keyword.KeywordIndex
    bm25: lexical.BM25Index


def load_release(
    prepared_manifest: Path,
    tracking_root: Path,
    model_version: int,
    dense_timeout_ms: int,
    destination: Path,
) -> LoadedRelease:
    database = (tracking_root / "tracking.db").resolve(strict=True)
    uri = f"sqlite:///{database}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    version = client.get_model_version(MODEL_NAME, str(model_version))
    if version.status != "READY" or not version.run_id:
        raise ValueError("the pinned model version must be ready and traceable")
    run = client.get_run(version.run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("gate.status") != "passed":
        raise ValueError("the pinned model version must come from a passing gate run")

    bundle = Path(
        client.download_artifacts(version.run_id, BUNDLE_ARTIFACT, str(destination))
    )
    data = load_validation(prepared_manifest)
    bundle_manifest = bundle / "manifest.json"
    manifest = check_bundle(bundle_manifest, data)
    for key in ("candidate_id", "dataset_version"):
        if (
            version.tags.get(key) != manifest[key]
            or run.data.tags.get(key) != manifest[key]
        ):
            raise ValueError(f"registry and model bundle disagree on {key}")
    source = manifest.get("source")
    commit = source.get("commit") if isinstance(source, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit
    ):
        raise ValueError("the pinned model must record its training Git commit")

    lexical_inputs = {
        "corpus": data.manifest_sha256,
        "tokenizer": sha256_file(Path(text.__file__)),
    }
    metadata = {
        "model_version": str(model_version),
        "candidate_id": manifest["candidate_id"],
        "training_commit": commit,
        "dataset_version": data.dataset_version,
        "prepared_manifest_sha256": data.manifest_sha256,
        "bundle_manifest_sha256": sha256_file(bundle_manifest),
        "keyword_version": manifest_digest(
            {**lexical_inputs, "keyword": sha256_file(Path(keyword.__file__))}
        ),
        "bm25_version": manifest_digest(
            {**lexical_inputs, "bm25": sha256_file(Path(lexical.__file__))}
        ),
        "dense_timeout_ms": dense_timeout_ms,
    }
    return LoadedRelease(
        Release(manifest_digest(metadata), **metadata),
        bundle,
        {document.document_id: document for document in data.documents},
        keyword.KeywordIndex(data.documents),
        lexical.BM25Index(data.documents),
    )
