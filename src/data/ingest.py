"""Fetch and inventory the pinned BEIR SciFact inputs."""

import argparse
import csv
import errno
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from pyarrow import parquet

from .integrity import manifest_digest, sha256_file


@dataclass(frozen=True)
class SourceFile:
    repository: str
    revision: str
    path: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repository}"
            f"/resolve/{self.revision}/{self.path}"
        )


SCIFACT_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
CORPUS_PATH = "corpus/corpus-00000-of-00001.parquet"
QUERIES_PATH = "queries/queries-00000-of-00001.parquet"
QRELS_COLUMNS = ("query-id", "corpus-id", "score")
SOURCES = (
    SourceFile(
        "BeIR/scifact",
        SCIFACT_REVISION,
        CORPUS_PATH,
        "243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45",
    ),
    SourceFile(
        "BeIR/scifact",
        SCIFACT_REVISION,
        QUERIES_PATH,
        "1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034",
    ),
    SourceFile(
        "BeIR/scifact-qrels",
        QRELS_REVISION,
        "train.tsv",
        "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
    ),
    SourceFile(
        "BeIR/scifact-qrels",
        QRELS_REVISION,
        "test.tsv",
        "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
    ),
)
DEFAULT_OUTPUT_ROOT = Path("data/raw/scifact")


def count_rows(path: Path) -> dict[str, int]:
    if path.suffix == ".parquet":
        return {"row_count": parquet.read_metadata(path).num_rows}

    with path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source, delimiter="\t")
        if rows.fieldnames != list(QRELS_COLUMNS):
            raise ValueError(f"unexpected relevance judgments header in {path.name}")

        row_count = 0
        judged_queries: set[str] = set()
        for row in rows:
            row_count += 1
            if row["query-id"]:
                judged_queries.add(row["query-id"])

    return {"row_count": row_count, "distinct_judged_queries": len(judged_queries)}


def ingest(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ingest-", dir=output_root) as temporary:
        staging = Path(temporary) / "snapshot"
        source_entries = []
        for source in SOURCES:
            path = staging / source.path
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with (
                    urlopen(source.url, timeout=30) as response,
                    path.open("wb") as target,
                ):
                    shutil.copyfileobj(response, target)
            except OSError as error:
                raise OSError(f"could not fetch {source.url}: {error}") from error

            digest = sha256_file(path)
            if digest != source.sha256:
                raise ValueError(
                    f"checksum mismatch for downloaded source: {source.path}"
                )

            source_entries.append(
                {
                    "repository": source.repository,
                    "revision": source.revision,
                    "path": source.path,
                    "sha256": digest,
                    **count_rows(path),
                }
            )

        inventory = {
            "dataset": "scifact",
            "manifest_version": 1,
            "sources": source_entries,
        }
        dataset_version = manifest_digest(inventory)
        manifest = {**inventory, "dataset_version": dataset_version}
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        (staging / "manifest.json").write_bytes(manifest_bytes)

        snapshot = output_root / dataset_version
        try:
            staging.rename(snapshot)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise

            if (snapshot / "manifest.json").read_bytes() != manifest_bytes:
                raise ValueError(
                    f"manifest mismatch in existing snapshot: {snapshot}"
                ) from error
            for source in SOURCES:
                path = snapshot / source.path
                if sha256_file(path) != source.sha256:
                    raise ValueError(
                        f"checksum mismatch in existing snapshot: {path}"
                    ) from error

    return snapshot / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="snapshot directory (default: %(default)s)",
    )
    args = parser.parse_args()
    try:
        manifest_path = ingest(args.output_root)
    except (OSError, ValueError, csv.Error) as error:
        print(f"SciFact ingestion failed: {error}", file=sys.stderr)
        return 1

    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
