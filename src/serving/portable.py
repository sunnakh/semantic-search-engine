"""Export a gate-approved SciFact release for offline serving."""

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from tempfile import TemporaryDirectory

from ..data.integrity import sha256_file
from ..data.prepared import DEFAULT_PROCESSED_ROOT, load_validation
from ..evaluation.gate import read_object
from ..registry.tracking import TRACKING_ROOT
from .release import (
    DEFAULT_DENSE_TIMEOUT_MS,
    LoadedRelease,
    load_model_bundle,
    load_release,
)

RELEASE_MANIFEST = "release.json"
SNAPSHOT_MANIFEST = Path("snapshot/manifest.json")


def _regular_hashes(root: Path) -> dict[str, str]:
    if not S_ISDIR(root.lstat().st_mode):
        raise ValueError(f"expected a regular artifact directory: {root}")
    hashes = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if S_ISREG(mode):
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        elif not S_ISDIR(mode):
            raise ValueError(f"expected a regular artifact file or directory: {path}")
    return hashes


def _verify_snapshot(manifest: Path) -> None:
    hashes = _regular_hashes(manifest.parent)
    if hashes.pop(manifest.name, None) is None:
        raise ValueError("prepared snapshot is missing its manifest")
    if hashes != read_object(manifest).get("artifact_sha256"):
        raise ValueError("prepared snapshot contains missing or unverified files")


def export_release(
    prepared_manifest: Path,
    tracking_root: Path,
    model_version: int,
    destination: Path,
    dense_timeout_ms: int = DEFAULT_DENSE_TIMEOUT_MS,
) -> Path:
    if model_version < 1 or dense_timeout_ms < 1:
        raise ValueError("model versions and timeouts must be positive")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"release destination already exists: {destination}")
    _verify_snapshot(prepared_manifest)
    if destination.resolve().is_relative_to(prepared_manifest.parent.resolve()):
        raise ValueError("release destination must be outside the prepared snapshot")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        copied_manifest = staging / SNAPSHOT_MANIFEST
        shutil.copytree(prepared_manifest.parent, copied_manifest.parent)
        (copied_manifest.parent / prepared_manifest.name).rename(copied_manifest)
        _verify_snapshot(copied_manifest)

        release = load_release(
            copied_manifest, tracking_root, model_version, dense_timeout_ms, staging
        )
        manifest = {
            "schema_version": 1,
            "registry_run_id": release.registry_run_id,
            "release": asdict(release.metadata),
            "artifact_sha256": _regular_hashes(staging),
        }
        (staging / RELEASE_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    return destination


def load_portable_release(root: Path) -> LoadedRelease:
    hashes = _regular_hashes(root)
    if hashes.pop(RELEASE_MANIFEST, None) is None:
        raise ValueError("portable release is missing its manifest")
    manifest = read_object(root / RELEASE_MANIFEST)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported portable release schema")
    if hashes != manifest.get("artifact_sha256"):
        raise ValueError("portable release checksum mismatch or missing files")

    metadata = manifest.get("release")
    if not isinstance(metadata, dict):
        raise ValueError("portable release metadata is missing")
    version = metadata.get("model_version")
    timeout = metadata.get("dense_timeout_ms")
    run_id = manifest.get("registry_run_id")
    if (
        not isinstance(version, str)
        or not re.fullmatch(r"[1-9][0-9]*", version)
        or type(timeout) is not int
        or timeout < 1
        or not isinstance(run_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", run_id)
    ):
        raise ValueError("invalid portable release version, timeout, or run ID")

    release = load_model_bundle(
        root / "bundle",
        load_validation(root / SNAPSHOT_MANIFEST),
        int(version),
        timeout,
        registry_run_id=run_id,
    )
    if asdict(release.metadata) != metadata:
        raise ValueError("portable release lineage does not match the runtime")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="prepared SciFact manifest")
    parser.add_argument("--tracking-root", type=Path, default=TRACKING_ROOT)
    parser.add_argument("--model-version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dense-timeout-ms", type=int, default=DEFAULT_DENSE_TIMEOUT_MS
    )
    args = parser.parse_args()
    prepared = args.manifest
    if prepared is None:
        manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
        if len(manifests) != 1:
            parser.error("run make prepare or select one snapshot with --manifest")
        (prepared,) = manifests

    exported = export_release(
        prepared,
        args.tracking_root,
        args.model_version,
        args.output,
        args.dense_timeout_ms,
    )
    print(exported)


if __name__ == "__main__":
    main()
