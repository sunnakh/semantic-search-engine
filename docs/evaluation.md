# SciFact retrieval evaluation

Run `make baselines` after `make prepare` to benchmark keyword overlap, TF-IDF, and BM25. Each method checks the prepared corpus and validation file hashes and writes `artifacts/evaluation/<dataset_version>/<method>-validation.json`. Each report includes the input manifest hash, per-query rankings and metrics, and macro averages across validation queries. `make baseline` runs keyword overlap alone. To select a method and snapshot, run `uv run --locked python -m src.evaluation.benchmark --method bm25 --manifest <path>`.

Run `make dense-baseline` to benchmark pretrained MiniLM. The first run installs the optional CPU inference packages and downloads the pinned model. Weights are cached under `artifacts/models/huggingface/`. After that, `HF_HUB_OFFLINE=1 UV_OFFLINE=1 make dense-baseline` evaluates from local files. The dense report records the model revision, inference settings, and runtime package versions.

## Retrieval methods

All three methods use the same title and abstract text, Unicode NFKC normalization, case folding, and word tokenization. They return positive scores in descending order and break ties by ascending document ID.

- **Keyword overlap** gives one point per distinct query token found in the document.
- **TF-IDF cosine** uses raw token counts, smoothed IDF `1 + ln((N + 1) / (df + 1))`, and L2 normalization of document and query vectors. `N` is the corpus size and `df` is the number of documents containing the term. Query terms outside the corpus vocabulary are omitted. [TF-IDF weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tf-idf-term-weighting)
- **BM25** uses one contribution per distinct query token and the positive IDF `ln(1 + (N - df + 0.5) / (df + 0.5))`. The default `k1 = 1.2` controls how quickly repeated occurrences lose incremental weight, and `b = 0.75` controls document length normalization. Both parameters were fixed before evaluation. [Lucene BM25Similarity](https://lucene.apache.org/core/9_12_3/core/org/apache/lucene/search/similarities/BM25Similarity.html)
- **Pretrained MiniLM** embeds each document's concatenated title and abstract and each validation claim into 384 dimensions. It uses CPU float32 inference, a batch size of 32, and a 256-token sequence limit. It L2-normalizes the vectors and computes exact cosine scores against the whole corpus, with ties broken by document ID. The encoder weights are frozen. [Pinned model revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

## Validation results

All methods ranked 5,183 abstracts for the same 162 validation queries in dataset `1ab7450c055a5e19db4d355b104845c7769e329ba9dca671b2f13b5f6524ea0c`.

| Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Queries with a relevant hit in top 10 |
|---|---:|---:|---:|---:|---:|
| Keyword overlap | 0.49017 | 0.51903 | 0.59979 | 0.46429 | 100 |
| TF-IDF cosine | 0.59845 | 0.65998 | 0.74228 | 0.56159 | 123 |
| BM25 | 0.65699 | 0.69599 | 0.75617 | 0.63419 | 125 |
| Pretrained MiniLM | 0.67623 | 0.76235 | 0.79938 | 0.64830 | 131 |
| Fine-tuned MiniLM | 0.70171 | 0.77160 | 0.83333 | 0.67146 | 136 |

BM25 improved mean nDCG@10 by 0.16682 over keyword overlap. Pretrained MiniLM improved mean nDCG@10 by a further 0.01924 over BM25. MiniLM scored higher than BM25 on 38 queries, lower on 24, and tied on 100.

nDCG@10 discounts each relevant hit by `1 / log2(rank + 1)` and divides by the ideal gain for that query. Recall divides the number of relevant hits within the cutoff by the number of judged relevant abstracts. MRR@10 credits only the first relevant hit. Unjudged results contribute zero gain under benchmark scoring. [BEIR evaluation implementation](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)

The official test benchmark is reserved for final evaluation. The prepared split audit records overlap between development and test claims that may affect interpretation of eventual test scores. [Data contract](data_contracts.md#prepared-snapshot)

The sequence limit truncates long inputs. MiniLM's published training mixture includes S2ORC scientific text, so overlap between its training text and SciFact documents is possible. The reported metrics measure retrieval quality on this validation split. [MiniLM training data and input limits](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#training-data)

## Fine-tuned candidate

Run `make train-candidate` after `make prepare`. The CPU run fine-tunes the pinned MiniLM encoder for one epoch with AdamW, a `1e-5` learning rate, batches of 16 judged pairs, a contrastive temperature of `0.05`, and seed `2026`. It excludes training IDs flagged by the prepared audit as exact normalized matches to official test claims. Validation and official test judgments are excluded from optimization.

For each query, the loss increases the softmax probability assigned to all known relevant documents in the batch. The remaining in-batch documents supply contrastive negatives. An unjudged pair can still be relevant, so this objective has a known false-negative risk.

The command saves and reloads the encoder, builds its corpus index, and reloads the index before scoring on validation. It publishes the encoder, index, comparison reports, and checksum manifest under `artifacts/models/candidates/<candidate_id>/`. The manifest records the prepared data version, training settings, base model revision, runtime versions, and source fingerprint. It records the Git commit when available.

The first run used 739 training pairs across 646 queries and took 47 optimizer steps. The saved candidate improved validation nDCG@10 by 0.02548 over frozen MiniLM. It scored higher on 21 queries, lower on 5, and tied on 136.
