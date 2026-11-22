"""Validate SciFact records and retain source locations for rejected rows."""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pyarrow import parquet

from .ingest import CORPUS_PATH, QRELS_COLUMNS, QUERIES_PATH


class Rejection(TypedDict):
    source: str
    row: int
    reason: str
    record: object


@dataclass
class ValidatedRecords:
    documents: dict[str, dict[str, str]]
    queries: dict[str, str]
    official_ids: dict[str, set[str]]
    qrels: dict[str, set[tuple[str, str]]]
    quarantine: list[Rejection]
    blockers: list[str]


def load_entities(
    raw_dir: Path, source: str, fields: tuple[str, ...], quarantine: list[Rejection]
) -> dict[str, dict[str, str]]:
    file = parquet.ParquetFile(raw_dir / source)
    missing = set(fields) - set(file.schema_arrow.names)
    if missing:
        raise ValueError(
            f"missing required columns in {source}: {', '.join(sorted(missing))}"
        )

    rows = file.read(columns=list(fields)).to_pylist()
    id_counts = Counter(row["_id"] for row in rows if isinstance(row["_id"], str))
    accepted: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, 1):
        entity_id, text = row["_id"], row["text"]
        if not isinstance(entity_id, str) or not entity_id.strip():
            reason = "invalid_id"
        elif id_counts[entity_id] > 1:
            reason = "duplicate_id"
        elif not isinstance(text, str) or not text.strip():
            reason = "blank_or_invalid_text"
        elif "title" in fields and not isinstance(row["title"], str):
            reason = "invalid_title"
        else:
            accepted[entity_id] = row
            continue

        quarantine.append(
            {"source": source, "row": row_number, "reason": reason, "record": row}
        )
    return accepted


def validate_records(raw_dir: Path) -> ValidatedRecords:
    quarantine: list[Rejection] = []
    blockers: list[str] = []
    documents = load_entities(
        raw_dir, CORPUS_PATH, ("_id", "title", "text"), quarantine
    )
    query_rows = load_entities(raw_dir, QUERIES_PATH, ("_id", "text"), quarantine)
    queries = {query_id: row["text"] for query_id, row in query_rows.items()}
    official_ids: dict[str, set[str]] = {"train": set(), "test": set()}
    qrels: dict[str, set[tuple[str, str]]] = {"train": set(), "test": set()}

    for split in ("train", "test"):
        source = f"{split}.tsv"
        rejection_count_before = len(quarantine)
        with (raw_dir / source).open(encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            if next(reader, None) != list(QRELS_COLUMNS):
                raise ValueError(f"unexpected relevance judgments header in {source}")

            for row in reader:
                if not row:
                    continue
                if row[0].strip():
                    official_ids[split].add(row[0])

                reason = qrel_rejection(row, queries, documents, qrels[split])
                if reason:
                    quarantine.append(
                        {
                            "source": source,
                            "row": reader.line_num,
                            "reason": reason,
                            "record": row,
                        }
                    )
                else:
                    qrels[split].add((row[0], row[1]))

        if split == "test" and len(quarantine) != rejection_count_before:
            blockers.append("quarantine would change the official test benchmark")

    overlap = sorted(official_ids["train"] & official_ids["test"])
    if overlap:
        blockers.append(
            f"query IDs appear in both official splits: {', '.join(overlap)}"
        )
    if not qrels["test"]:
        blockers.append("the official test benchmark has no valid judgments")
    return ValidatedRecords(
        documents, queries, official_ids, qrels, quarantine, blockers
    )


def qrel_rejection(
    row: list[str],
    queries: dict[str, str],
    documents: dict[str, dict[str, str]],
    seen: set[tuple[str, str]],
) -> str | None:
    if len(row) != len(QRELS_COLUMNS):
        return "invalid_column_count"
    query_id, document_id, score = row
    if not query_id.strip() or not document_id.strip():
        return "invalid_id"
    try:
        relevance = int(score)
    except ValueError:
        return "invalid_relevance"
    if relevance != 1:
        return "invalid_relevance"
    if query_id not in queries:
        return "unknown_query_id"
    if document_id not in documents:
        return "unknown_document_id"
    if (query_id, document_id) in seen:
        return "duplicate_qrel"
    return None
