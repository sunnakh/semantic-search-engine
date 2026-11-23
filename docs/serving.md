# SciFact search API

From a checked-out project commit, run `make prepare` and `make pipeline`. Use the `registered_version` printed by the passing pipeline run. The server requires a recorded training Git commit, so regenerate any model trained from a source archive without Git metadata.

```sh
make serve SERVE_ARGS='--model-version 1'
```

Pass `--manifest` to select a specific prepared snapshot and `--tracking-root` to select a local MLflow store. Defaults are the sole prepared SciFact snapshot and `artifacts/mlflow`. The API binds to `127.0.0.1:8000` by default.

```sh
curl -fsS http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"exercise and heart health","method":"dense","top_k":5}'
```

`POST /api/v1/search` accepts `dense`, `bm25`, or `keyword`. The query must contain 1–512 characters after trimming, `top_k` must be an integer from 1 to 50, and the body may be at most 8 KiB. Results include the actual method and version, a release ID, a request ID, and a fallback flag. `GET /api/v1/models` resolves the release ID to its data, model bundle, training commit, and lexical index versions. `GET /api/v1/documents/{id}` returns document metadata.

Startup verifies the pinned MLflow version, its passing gate run, the model bundle, and the matching prepared corpus before reporting ready. The API builds lexical indexes in memory from that verified snapshot. Dense inference runs in a separate CPU process with a configurable deadline that includes queue time. When an in-flight request exceeds the deadline, the API kills the worker and uses BM25. When the worker exits, liveness stays healthy, readiness returns 503, and search requests can still use BM25. Restart the API to reload the pinned model.

`/metrics` exports search latency, fallback counts, and dense-worker readiness. Prediction logs contain document IDs, a keyed query hash, query length, release lineage, and latency. To correlate hashes across restarts, set `SCIFACT_QUERY_HASH_KEY` to a secret shared by the intended instances. Otherwise, each instance generates its own key at startup.

## Local verification

With the server running, set `SCIFACT_MANIFEST` to the path of the same manifest used at startup.

```sh
uv run --locked python -m src.serving.load_test \
  --manifest "$SCIFACT_MANIFEST" \
  --users 5 --requests 150
```

On an Apple M4 Pro with 14 CPU cores and 24 GB RAM, 150 dense searches across 5 concurrent loopback clients produced a server-side p95 of **25.315 ms** against the **500 ms** target. Loopback client p95 was 27.166 ms and throughput was 228.3 requests/s. Ten HTTP searches warmed the service before measurement. Server latency excludes network time and response serialization. The API and worker had observed RSS of approximately 324 MiB and 530 MiB respectively. The run recorded zero HTTP errors and zero fallbacks.

For a local failure drill, identify the test server's `src.serving.worker` child process and send it `SIGKILL`. Repeat the same search with `method` set to `dense` and `bm25`. The drill returned matching BM25 results, marked the dense request as a fallback, and left liveness at 200 and readiness at 503. Restart the test server after the drill.
