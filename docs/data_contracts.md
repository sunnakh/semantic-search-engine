# SciFact data contract

See the [project specification](requirements.md#4-dataset-specification) for the dataset and evaluation requirements.

## Source files

| Input | Pinned source revision | File | Fields used |
|---|---|---|---|
| Corpus | [BeIR/scifact at `b3b5335`](https://huggingface.co/datasets/BeIR/scifact/tree/b3b5335604bf5ee3c4447671af975ea25143d4f5) | `corpus/corpus-00000-of-00001.parquet` | `_id`, `title`, `text` |
| Queries | [BeIR/scifact at `b3b5335`](https://huggingface.co/datasets/BeIR/scifact/tree/b3b5335604bf5ee3c4447671af975ea25143d4f5) | `queries/queries-00000-of-00001.parquet` | `_id`, `text` |
| Relevance judgments | [BeIR/scifact-qrels at `2938d17`](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029) | `train.tsv`, `test.tsv` | `query-id`, `corpus-id`, `score` |

The dataset card reports 5,183 corpus rows and 1,109 query rows. A relevance judgment is one query–document pair, so the ingestion report must count judgment rows and distinct judged queries separately. [BEIR SciFact dataset card](https://huggingface.co/datasets/BeIR/scifact#scifact-data-splits)

From the project root, run `make ingest` with `uv` installed. The command verifies each download against its pinned SHA-256 and prints the path to `data/raw/scifact/<dataset_version>/manifest.json`. The manifest records source revisions, hashes, row counts, and distinct judged-query counts for the two relevance files. A rerun fetches the sources again and returns the same snapshot if the published files and manifest still match.

## Normalized records

| Record | Fields | Required checks |
|---|---|---|
| Document | `document_id`, `title`, `text`, `dataset_version` | Unique nonempty ID, string title (empty allowed), and nonblank abstract text |
| Query | `query_id`, `text`, `split` | Unique nonempty ID, nonblank text, and split in `train`, `validation`, or `test` |
| Relevance judgment | `query_id`, `document_id`, `relevance`, `split` | Unique query–document pair, both IDs resolve, integer relevance is `1` for this source, and split matches the query |

Treat IDs as strings across sources. Parse the TSV ID columns as text so numeric-looking identifiers retain their exact representation. Parse the score separately as an integer. Keep the source files immutable and record their repository, full revision, relative path, SHA-256, and row count in the ingestion manifest.

## Split and rejection policy

- Derive official membership from `train.tsv` and `test.tsv`. Preserve the official test query IDs. Assign validation membership deterministically from official training queries and keep each query's judgments together.
- Report source queries with no judgments separately. Keep them out of supervised training and benchmark scoring. Missing judgments supply no verified negative labels.
- Quarantine malformed or duplicate records and unresolved judgments with their source file, row location, and a rejection reason. Fail validation if a query is assigned to both official splits or quarantine would remove a query or judgment from the official test benchmark.
- Normalize claim text with Unicode NFKC, case folding, and word tokenization for the overlap audit. Flag identical token sequences and pairs whose word-bigram Jaccard similarity is at least 0.80. This lexical check can miss paraphrases and flag claims with different meanings.
- Keep connected groups of flagged development queries together. Sort groups using SHA-256 and seed `2026`, then select a prefix closest to 20% of judged development queries for validation. Leave at least one group in each development split. Preserve the official test set and report matches against it for review.

## Prepared snapshot

Run `make prepare` after `make ingest`. Preparation reads the local raw snapshot and verifies its source hashes. If multiple raw snapshots exist, select one with `uv run --locked python -m src.data.prepare --raw-manifest <path>`.

The command prints the path to `data/processed/scifact/<dataset_version>/manifest.json`. The manifest records raw lineage, the transformation version, split policy, output hashes, and record counts. The snapshot contains `documents.parquet` and query and judgment Parquet files under `queries/split=<split>/` and `qrels/split=<split>/`. The directory partition supplies the `split` column. The corpus fits in one file and is shared by all three splits.

`quarantine.jsonl` records source locations and rejection reasons. `split_audit.json` reports excluded query IDs and potential claim overlap. When row validation would change the official test benchmark or finds a query in both official splits, preparation exits with an error and writes diagnostics under `data/processed/scifact/rejected/`.

On the pinned snapshot, preparation accepted 5,183 documents and all 1,258 judgments. It produced 647 training, 162 validation, and 300 test queries. The audit flagged 18 development/test pairs. Two are exact after normalization, with one pair across train/test and one across validation/test. The other 16 are lexical near-matches. Training or tuning on a matching claim can make the test score optimistic, so record this overlap when reporting benchmark results.

## First pipeline acceptance checks

- The ingestion report includes actual corpus and query counts, judgment counts per source split, and distinct judged query counts per source split.
- Fixtures for blank text, a duplicate ID, an unknown document reference, and an invalid score produce the expected quarantine reasons. An official split overlap fails validation.
- On the pinned source files, each accepted judgment joins to exactly one query and one document. The official test query IDs and judgments match the source TSV.
- On a rerun with identical inputs and configuration, split membership and the hashes of the prepared artifacts match.
