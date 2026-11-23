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

## Offline gate and MLflow registration

Run `make pipeline` after `make prepare` to train a candidate, verify its bundle, and record the gate decision. The gate checks artifact hashes, the encoder and index dimensions, the indexed corpus, and the validation queries and judgments. It recomputes metrics from the stored rankings. To evaluate an existing bundle, run `uv run --locked --extra dense --extra registry python -m src.training.pipeline --candidate <candidate-manifest>`.

The reference is the registered `champion` alias when available and frozen MiniLM otherwise. The nDCG@10 floor is the higher score from BM25 and the reference. The Recall@5 floor uses the higher reference or BM25 score with a 0.02 absolute tolerance. A reference report from a different prepared snapshot requires reevaluation before it can be compared.

MLflow stores each attempt in the `scifact-candidates` experiment in `artifacts/mlflow/tracking.db`. Every gate decision records its result and reasons. Passing bundles are registered under `scifact-dense-retriever` with `status=candidate`. Repeating a gate attempt logs another run and reuses the registered version for the same candidate ID. Each registered model version contains an MLflow `MLmodel` descriptor and the exact saved encoder, index, manifest, and reports under `bundle/`. [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

Point MLflow clients at the local SQLite tracking and registry URI before resolving `models:/scifact-dense-retriever/1`. This version was registered at 0.70171 nDCG@10 and 0.77160 Recall@5. For a deliberate failure injection, negating the saved document vectors produced zero nDCG@10 and Recall@5 on the same validation split. The gate logged the rejection in MLflow and the registry stayed at one version.

The CLI exits with `2` for a gate rejection and `1` for an execution error. Readiness, latency, shadow traffic, and activation are checked in the release and serving stages.

## Fine-tune the Qwen embedding model with LoRA

From the project directory, run `make ingest && make prepare` if you need the prepared SciFact snapshot. Then run:

```sh
make train-qwen-embedding
```

This uses **LoRA** on the unquantized, pinned `Qwen/Qwen3-Embedding-0.6B` checkpoint. The base weights stay frozen in FP32. Training updates rank-8 adapters on the attention query and value projections. Defaults include one epoch, 8 claim–paper pairs per batch, a learning rate of `5e-5`, and Apple GPU acceleration when available. The first run may download approximately 1.1 GiB of model weights.

For a short training check, use `make train-qwen-embedding QWEN_ARGS='--max-steps 2'`. For a full run, omit `--max-steps`. Use `--batch-size 4` if the default batch exceeds available memory.

The command prints the path to the saved adapter's manifest. Pass that path to evaluation:

```sh
make evaluate-qwen-embedding QWEN_ARGS='--run artifacts/models/qwen-lora/REPLACE_WITH_RUN_ID/manifest.json'
```

Evaluation reloads the adapter and re-embeds the complete validation corpus. It compares nDCG@10 and Recall@5 with frozen Qwen using the same prepared snapshot, query instruction, and inference settings. It reuses a matching frozen report when present and builds one otherwise. If an existing report was generated with other settings, pass a new path with `--frozen-report`. Results go in `validation.json` and `comparison.json` beside the adapter manifest. This is a local experiment and the official test split stays reserved.

## Qwen3.5-9B MLX reranker

On Apple Silicon, use the Qwen3.5-9B MLX 4-bit checkpoint downloaded through LM Studio. The default model directory is `~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit`. Pass `--model` in `QWEN_ARGS` to use a different location for the same checkpoint.

After `make prepare`, run a short smoke test or train for a full epoch:

```sh
make train-qwen QWEN_ARGS='--max-steps 2'
make train-qwen QWEN_ARGS='--epochs 1'
```

Plain `make train-qwen` runs 100 steps. The 4-bit weights stay frozen, and rank-8 adapters update 139,264 parameters on the last attention block's query and value projections. Each training group supplies a judged relevant paper and a BM25 hard negative. The command prints the step loss and epoch summaries and saves the adapter, checksums, and `epoch_history` under `artifacts/models/qwen-9b-qlora/`.

Set `SCIFACT_RUN` to the manifest path printed by training and evaluate the same BM25 shortlist with and without the adapter:

```sh
make evaluate-qwen QWEN_ARGS="--run $SCIFACT_RUN"
```

The full validation run across 162 claims improved nDCG@10 from 0.6763 for frozen Qwen to 0.6986 with the two-epoch adapter. Recall@5 rose from 0.7377 to 0.7438. BM25 supplied the top-10 shortlist in both cases.

To compare embedding retrieval alone with embedding retrieval followed by 9B reranking, reuse the frozen Qwen3-Embedding-0.6B validation report:

```sh
SCIFACT_CANDIDATES=artifacts/evaluation/qwen3-embedding-0.6b/qwen-validation.json
make evaluate-qwen QWEN_ARGS="--run $SCIFACT_RUN --candidate-report $SCIFACT_CANDIDATES --query-limit 5"
```

The embedding evaluation workflow above generates the report if it is missing. Omit `--query-limit 5` to evaluate all claims. Hybrid results are written beside the adapter manifest in `comparison-qwen-embedding-validation.json` for the full run. The hybrid workflow was verified on one validation claim.
