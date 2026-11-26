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

Official test scores are reported separately below. The prepared split audit records overlap between development and test claims. [Data contract](data_contracts.md#prepared-snapshot)

The sequence limit truncates long inputs. MiniLM's published training mixture includes S2ORC scientific text, so overlap between its training text and SciFact documents is possible. The reported metrics measure retrieval quality on this validation split. [MiniLM training data and input limits](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#training-data)

## Official test results

After selecting a registered model using development evidence, evaluate it without training or registering another model:

```sh
make test-benchmark TEST_BENCHMARK_ARGS='--model-version 3'
```

Use the version printed by your own pipeline when reproducing the workflow; version numbers belong to each local registry. Optional `--manifest`, `--tracking-root`, and `--output` select the snapshot, registry, and evidence directory. The default directory is `artifacts/evaluation/official-test/model-<version>/`. The command writes `selection.json` before opening test labels, then `report.json` with each method's per-query rankings, metrics, lineage, runtime versions, and overlap sensitivity analysis. It validates input and bundle checksums, uses four CPU threads, and performs no optimizer steps or registry writes. Training and registration continue to accept only development inputs.

Identical reruns preserve the evidence files. Different selections or reports cannot overwrite them; use a new `--output` directory when changing the code, model, or snapshot. Subsequent experiments informed by these scores are test-informed development and need a new untouched evaluation set for an independent final claim.

Measured locally on November 26, 2026: the model version **3** selected before this run ranked all **5,183 abstracts for 300 official test claims** in the same prepared snapshot used above. BM25 parameters, the frozen MiniLM revision, and the saved candidate weights were unchanged.

| Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Queries with a relevant hit in top 10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.66168 | 0.72594 | 0.79094 | 0.62762 | 243 |
| Frozen MiniLM | 0.64508 | 0.73794 | 0.78333 | 0.60472 | 238 |
| Fine-tuned MiniLM, version 3 | 0.66611 | 0.74672 | 0.82222 | 0.62111 | 250 |

Fine-tuning improved MiniLM's test nDCG@10 by **0.02103** and Recall@5 by **0.00878**. Its nDCG@10 exceeded BM25 by only **0.00443**, while MRR@10 was lower. Against BM25, it improved nDCG on 72 queries, regressed on 70, and tied on 158. These point estimates do not establish a statistically significant or general improvement over BM25. Qwen experiments were not included in this final test run.

The existing audit flags **18 development/test pairs involving 18 distinct test claims**: two exact normalized matches and 16 near duplicates. Training excludes the development training claim that exactly matches a test claim; an exact match in validation and the near duplicates remain interpretation risks. Excluding all 18 flagged test claims gives this supplementary **282-query** comparison, scored from the same saved rankings:

| Method | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.65684 | 0.72352 | 0.79178 | 0.62217 |
| Frozen MiniLM | 0.63663 | 0.72920 | 0.77660 | 0.59663 |
| Fine-tuned MiniLM, version 3 | 0.65318 | 0.73499 | 0.81442 | 0.60727 |

On this filtered subset, fine-tuning still improves MiniLM, but nDCG@10 is **0.00365 below BM25**. This is a sensitivity check, not a replacement official score or proof that all overlap was removed. The audit covers claim similarity; pretraining overlap with scientific abstracts remains possible.

The measured candidate is `4ce988ac1b3dcd29d345553ab88fa0cb1f25f62b09ce6782b905b28cdc2d8902`, trained at commit `9edd3b86205c2099f4eb7a9f0902a21669dbc0f3`, registered by run `b644b0bd4cd944f08e545f70e82e3693`. Its bundle manifest SHA-256 is `0eb3463a4bc5354023f5034ce364ff4e1b8ac26f07f920e680da3ce3008267e9`. The evaluation records the parent Git commit plus the working source fingerprint (`c11eab77746b1a1ec1a9a77ca5b73cdbd3836567a9c2cbdd2a07f93ba30f234a`) because the benchmark implementation was not yet committed. The local report SHA-256 is `7c891f6c149f46157045e158385bfe1f2592532f51ccd792a1b89951a6859f84`; generated reports and checkpoints stay outside version control.

## Fine-tuned candidate

Run `make train-candidate` after `make prepare`. The CPU run fine-tunes the pinned MiniLM encoder for one epoch with AdamW, a `1e-5` learning rate, batches of 16 judged pairs, a contrastive temperature of `0.05`, and seed `2026`. It excludes training IDs flagged by the prepared audit as exact normalized matches to official test claims. Validation and official test judgments are excluded from optimization.

For each query, the loss increases the softmax probability assigned to all known relevant documents in the batch. The remaining in-batch documents supply contrastive negatives. An unjudged pair can still be relevant, so this objective has a known false-negative risk.

The command saves and reloads the encoder, builds its corpus index, and reloads the index before scoring on validation. It publishes the encoder, index, comparison reports, and checksum manifest under `artifacts/models/candidates/<candidate_id>/`. The manifest records the prepared data version, training settings, base model revision, runtime versions, and source fingerprint. It records the Git commit when available.

The first run used 739 training pairs across 646 queries and took 47 optimizer steps. The saved candidate improved validation nDCG@10 by 0.02548 over frozen MiniLM. It scored higher on 21 queries, lower on 5, and tied on 136.

## Offline gate and MLflow registration

Run `make pipeline` after `make prepare` to train a candidate, verify its bundle, and record the gate decision. The gate checks artifact hashes, the encoder and index dimensions, the indexed corpus, and the validation queries and judgments. It recomputes metrics from the stored rankings. To evaluate an existing bundle, run `uv run --locked --extra dense --extra registry python -m src.training.pipeline --candidate <candidate-manifest>`.

For candidate evaluation against a running service, select its API explicitly:

```sh
make pipeline PIPELINE_ARGS='--champion-api-url http://127.0.0.1:8000'
```

The gate reads `/api/v1/models`, resolves that active model version from the selected MLflow registry, and verifies the full release lineage and matching prepared snapshot. Its saved validation report supplies the champion metrics. The active worker determines the reference even if the release-state file or MLflow alias points to another version. After promotion or rollback, restart the API before evaluating against the newly selected release. Portable APIs work with the same command when their source model version is available in the selected registry.

The gate rejects registration if the API is unavailable, returns invalid metadata, disagrees with the registry, or changes its active release during evaluation. MLflow records `gate.champion_release_id`, `gate.champion_model_version`, and `gate.champion_run_id`. This check captures the observed champion for this evaluation. A deployment can change after the check, so promotion still requires a separate release decision.

Without `--champion-api-url`, the existing offline workflow uses the registered `champion` alias when available and frozen MiniLM otherwise. Use this for the initial model before a service exists. For all modes, the nDCG@10 floor is the higher score from BM25 and the reference. The Recall@5 floor uses the higher reference or BM25 score with a 0.02 absolute tolerance. A reference report from a different prepared snapshot requires reevaluation before it can be compared.

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

See the [QLoRA workflow](../README.md#fine-tune-qwen35-9b-and-watch-training) for training commands, evaluation, and measured results.
