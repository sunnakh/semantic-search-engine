"""Checksums for dataset artifacts and canonical manifest contents."""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def manifest_digest(inventory: dict[str, object]) -> str:
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_hashes(directory: Path) -> dict[str, str]:
    return {
        file.relative_to(directory).as_posix(): sha256_file(file)
        for file in sorted(directory.rglob("*"))
        if file.is_file()
    }
