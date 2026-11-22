"""Prepare a versioned SciFact dataset from a verified raw snapshot."""

import argparse
import csv
import errno
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
from pyarrow import parquet

from .ingest import DEFAULT_OUTPUT_ROOT, SOURCES
from .integrity import file_hashes, manifest_digest, sha256_file
from .prepared import DEFAULT_PROCESSED_ROOT, DOCUMENTS_FILE, PART_FILE
from .splits import SPLIT_POLICY, SPLITS, assign_splits, audit_claims
from .validate import ValidatedRecords, validate_records

TRANSFORM_VERSION = 1
DOCUMENT_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("dataset_version", pa.string()),
    ]
)
QUERY_SCHEMA = pa.schema([("query_id", pa.string()), ("text", pa.string())])
QREL_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("document_id", pa.string()),
        ("relevance", pa.int8()),
    ]
)


def verify_raw_manifest(path: Path) -> tuple[str, list[dict[str, object]]]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict):
        raise TypeError("the raw manifest must contain a JSON object")
    dataset_version = inventory.pop("dataset_version", None)
    if (
        inventory.get("dataset") != "scifact"
        or inventory.get("manifest_version") != 1
        or not isinstance(dataset_version, str)
        or dataset_version != manifest_digest(inventory)
    ):
        raise ValueError("invalid SciFact raw manifest or dataset version")

    entries = inventory.get("sources")
    if not isinstance(entries, list) or len(entries) != len(SOURCES):
        raise ValueError("raw manifest does not list the pinned SciFact files")
    for entry, source in zip(entries, SOURCES, strict=True):
        if not isinstance(entry, dict) or any(
            entry.get(field) != getattr(source, field)
            for field in ("repository", "revision", "path", "sha256")
        ):
            raise ValueError(f"raw manifest does not match pinned source {source.path}")
        if sha256_file(path.parent / source.path) != source.sha256:
            raise ValueError(f"raw file checksum mismatch: {source.path}")
    return dataset_version, entries


def write_dataset(
    staging: Path, records: ValidatedRecords, membership: dict[str, str], version: str
) -> dict[str, dict[str, int]]:
    documents = [
        {
            "document_id": document_id,
            "title": row["title"],
            "text": row["text"],
            "dataset_version": version,
        }
        for document_id, row in sorted(records.documents.items())
    ]
    parquet.write_table(
        pa.Table.from_pylist(documents, DOCUMENT_SCHEMA), staging / DOCUMENTS_FILE
    )

    counts = {"documents": {"total": len(documents)}, "queries": {}, "qrels": {}}
    for split in SPLITS:
        source_split = "test" if split == "test" else "train"
        queries = [
            {"query_id": query_id, "text": records.queries[query_id]}
            for query_id in sorted(membership)
            if membership[query_id] == split
        ]
        qrels = [
            {"query_id": query_id, "document_id": document_id, "relevance": 1}
            for query_id, document_id in sorted(records.qrels[source_split])
            if membership[query_id] == split
        ]
        for name, rows, schema in (
            ("queries", queries, QUERY_SCHEMA),
            ("qrels", qrels, QREL_SCHEMA),
        ):
            partition = staging / name / f"split={split}"
            partition.mkdir(parents=True)
            parquet.write_table(
                pa.Table.from_pylist(rows, schema), partition / PART_FILE
            )
            counts[name][split] = len(rows)
    return counts


def prepare(raw_manifest: Path, output_root: Path) -> Path:
    raw_version, sources = verify_raw_manifest(raw_manifest)
    records = validate_records(raw_manifest.parent)
    train_ids = {query_id for query_id, _ in records.qrels["train"]}
    test_ids = {query_id for query_id, _ in records.qrels["test"]}
    official_ids = records.official_ids["train"] | records.official_ids["test"]

    matches = []
    membership: dict[str, str] = {}
    if not records.blockers:
        matches = audit_claims(
            {query_id: records.queries[query_id] for query_id in train_ids | test_ids},
            test_ids,
        )
        try:
            membership = assign_splits(train_ids, test_ids, matches)
        except ValueError as error:
            records.blockers.append(str(error))
    audit = {
        "unjudged_query_ids": sorted(records.queries.keys() - official_ids),
        "query_ids_without_valid_judgments": sorted(
            (records.queries.keys() & official_ids) - train_ids - test_ids
        ),
        "development_matches": [
            pair
            for pair in matches
            if pair["left_query_id"] in train_ids
            and pair["right_query_id"] in train_ids
        ],
        "official_test_matches": [
            pair
            for pair in matches
            if pair["left_query_id"] in test_ids or pair["right_query_id"] in test_ids
        ],
    }
    lineage = {
        "raw_dataset_version": raw_version,
        "transform_version": TRANSFORM_VERSION,
        "split_policy": SPLIT_POLICY,
        "pyarrow_version": pa.__version__,
    }
    version = manifest_digest(lineage)
    versions = output_root / "rejected" if records.blockers else output_root
    versions.mkdir(parents=True, exist_ok=True)
    snapshot = versions / version
    with TemporaryDirectory(prefix=".prepare-", dir=versions) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        counts = (
            {}
            if records.blockers
            else write_dataset(staging, records, membership, version)
        )
        (staging / "split_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (staging / "quarantine.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as report:
            for row in records.quarantine:
                report.write(json.dumps(row, sort_keys=True, default=str) + "\n")

        manifest = {
            **lineage,
            "dataset_version": version,
            "input_sources": sources,
            "status": "blocked" if records.blockers else "ready",
            "blockers": records.blockers,
            "record_counts": counts,
            "quarantine_count": len(records.quarantine),
            "artifact_sha256": file_hashes(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            staging.rename(snapshot)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if file_hashes(snapshot) != file_hashes(staging):
                raise ValueError(
                    f"prepared snapshot differs on rerun: {snapshot}"
                ) from error

    manifest_path = snapshot / "manifest.json"
    if records.blockers:
        raise ValueError(f"{'; '.join(records.blockers)}. Diagnostics: {manifest_path}")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        help="manifest selected from the local raw snapshots",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    args = parser.parse_args()
    try:
        raw_manifest = args.raw_manifest
        if raw_manifest is None:
            manifests = sorted(DEFAULT_OUTPUT_ROOT.glob("*/manifest.json"))
            if len(manifests) != 1:
                raise ValueError(
                    "run make ingest for a raw snapshot or select one with --raw-manifest"
                )
            (raw_manifest,) = manifests
        print(prepare(raw_manifest, args.output_root))
    except (OSError, TypeError, ValueError, csv.Error) as error:
        print(f"SciFact preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
