# Model card · SciFact MiniLM v3

This release ranks scientific abstracts by relevance to a claim or search query. It helps researchers decide which papers to read first. Its cosine scores are ranking scores. Assessing whether a paper supports or refutes a claim requires reading the evidence. The measured scope is the SciFact abstract corpus.

## Model and data

| Item | Details |
|---|---|
| Registry release | `scifact-dense-retriever` version `3` in the local MLflow registry |
| Base encoder | `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Ranking | Encode the query and each title plus abstract as normalized 384-dimensional vectors, then rank by exact cosine similarity |
| Input limit | 256 tokens per input |
| Prepared snapshot | `1ab7450c055a5e19db4d355b104845c7769e329ba9dca671b2f13b5f6524ea0c` |
| Training commit | `9edd3b86205c2099f4eb7a9f0902a21669dbc0f3` |
| MLflow run | `b644b0bd4cd944f08e545f70e82e3693` |

Training used one epoch of contrastive fine-tuning over judged SciFact claim–paper pairs, with AdamW at a learning rate of `1e-5`, batch size 16, temperature 0.05, and seed 2026. The training workflow excludes training claims that exactly match official test claims after normalization. Validation guides registration. See the [dataset specification](requirements.md#4-dataset-specification).

## Ranking quality

| Evaluation set | BM25 nDCG@10 | Frozen MiniLM nDCG@10 | Fine-tuned MiniLM nDCG@10 | Fine-tuned MiniLM Recall@5 |
|---|---:|---:|---:|---:|
| Validation (162 claims) | 0.65699 | 0.67623 | 0.70171 | 0.77160 |
| Official test (300 claims) | 0.66168 | 0.64508 | 0.66611 | 0.74672 |
| Test excluding flagged overlap (282 claims) | 0.65684 | 0.63663 | 0.65318 | 0.73499 |

Fine-tuning improved MiniLM across both test comparisons. The ordering against BM25 changed when 18 claims flagged for development overlap were excluded. The full test also gave BM25 a higher MRR@10 (0.62762 versus 0.62111). These measurements leave the overall ranking advantage over BM25 uncertain.

## Limits and operations

- Long abstracts can be truncated. Incomplete relevance judgments can leave useful results uncredited and introduce false negatives during training.
- The split audit detects some overlapping claims. Pretraining overlap with scientific abstracts remains possible. The benchmark covers SciFact, and researcher time savings and generalization to other corpora remain unmeasured.
- The API pins the model and matching snapshot. If dense inference is unavailable or times out, it falls back to BM25. Use the response's method and fallback flag to identify the retrieval path, and `/api/v1/models` to inspect release lineage.
- The local MiniLM serving benchmark measured a 25.315 ms warm server-side p95 across 150 requests and five concurrent loopback clients, against a 500 ms target. A separate local drill verified BM25 fallback after worker failure. Cloud serving performance remains unmeasured. See the [serving guide](serving.md#local-verification) and [monitoring runbook](monitoring.md).
