"""Layout and verified reads for prepared SciFact snapshots."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pyarrow import parquet

from ..documents import Document
from .integrity import sha256_file

DEFAULT_PROCESSED_ROOT = Path("data/processed/scifact")
DOCUMENTS_FILE = "documents.parquet"
PART_FILE = "part-00000.parquet"
VALIDATION_QUERIES = f"queries/split=validation/{PART_FILE}"
VALIDATION_QRELS = f"qrels/split=validation/{PART_FILE}"
TRAIN_QUERIES = f"queries/split=train/{PART_FILE}"
TRAIN_QRELS = f"qrels/split=train/{PART_FILE}"
SPLIT_AUDIT = "split_audit.json"


@dataclass
class PreparedSplit:
    dataset_version: str
    manifest_sha256: str
    documents: list[Document]
    queries: dict[str, str]
    relevant_ids: dict[str, set[str]]
    exact_test_overlap_ids: tuple[str, ...] = ()


def load_validation(manifest_path: Path) -> PreparedSplit:
    return load_split(manifest_path, "validation")


def load_split(
    manifest_path: Path, split: Literal["train", "validation"]
) -> PreparedSplit:
    if split not in ("train", "validation"):
        raise ValueError("candidate workflows support only train and validation splits")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "ready":
        raise ValueError("a ready prepared dataset manifest is required")

    version = manifest.get("dataset_version")
    if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("invalid prepared dataset version")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise TypeError("prepared manifest must contain an artifact checksum mapping")

    snapshot = manifest_path.parent
    query_file, qrel_file = (
        (TRAIN_QUERIES, TRAIN_QRELS)
        if split == "train"
        else (VALIDATION_QUERIES, VALIDATION_QRELS)
    )
    inputs = [DOCUMENTS_FILE, query_file, qrel_file]
    if split == "train":
        inputs.append(SPLIT_AUDIT)
    for name in inputs:
        if hashes.get(name) != sha256_file(snapshot / name):
            raise ValueError(f"prepared artifact checksum mismatch: {name}")

    document_rows = parquet.ParquetFile(snapshot / DOCUMENTS_FILE).read(
        columns=["document_id", "title", "text"]
    )
    documents = [Document(**row) for row in document_rows.to_pylist()]
    query_rows = parquet.ParquetFile(snapshot / query_file).read(
        columns=["query_id", "text"]
    )
    queries = {row["query_id"]: row["text"] for row in query_rows.to_pylist()}
    qrels = parquet.ParquetFile(snapshot / qrel_file).read().to_pylist()
    relevant_ids: dict[str, set[str]] = defaultdict(set)
    corpus_ids = {document.document_id for document in documents}
    for row in qrels:
        if row["relevance"] != 1 or row["document_id"] not in corpus_ids:
            raise ValueError(
                f"{split} judgments must reference relevant corpus documents"
            )
        relevant_ids[row["query_id"]].add(row["document_id"])

    if not queries or queries.keys() != relevant_ids.keys():
        raise ValueError(
            f"{split} queries and judgments must cover the same nonempty query set"
        )
    overlap: set[str] = set()
    if split == "train":
        audit = json.loads((snapshot / SPLIT_AUDIT).read_text(encoding="utf-8"))
        for pair in audit["official_test_matches"]:
            if pair["match"] == "exact_normalized":
                overlap.update(
                    queries.keys() & {pair["left_query_id"], pair["right_query_id"]}
                )
    return PreparedSplit(
        version,
        hashlib.sha256(manifest_bytes).hexdigest(),
        documents,
        queries,
        relevant_ids,
        tuple(sorted(overlap)),
    )
