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

## Browser search UI

With the API running, open a second terminal in the same project directory.

```sh
make ui
```

Open [SciFact Search](http://127.0.0.1:8501). Enter a query, choose a retrieval method or compare dense and BM25 side by side, and click **Search**. Results include the rank, paper title, method-specific score, and expandable abstract. The interface shows when the API falls back to BM25.

The UI calls `http://127.0.0.1:8000` by default. Set `SCIFACT_API_URL` when the API uses a different base URL. To change the UI port, run `make ui UI_ARGS='--server.port 8502'`.

## Local verification

With the server running, set `SCIFACT_MANIFEST` to the path of the same manifest used at startup.

```sh
uv run --locked python -m src.serving.load_test \
  --manifest "$SCIFACT_MANIFEST" \
  --users 5 --requests 150
```

On an Apple M4 Pro with 14 CPU cores and 24 GB RAM, 150 dense searches across 5 concurrent loopback clients produced a server-side p95 of **25.315 ms** against the **500 ms** target. Loopback client p95 was 27.166 ms and throughput was 228.3 requests/s. Ten HTTP searches warmed the service before measurement. Server latency excludes network time and response serialization. The API and worker had observed RSS of approximately 324 MiB and 530 MiB respectively. The run recorded zero HTTP errors and zero fallbacks.

For a local failure drill, identify the test server's `src.serving.worker` child process and send it `SIGKILL`. Repeat the same search with `method` set to `dense` and `bm25`. The drill returned matching BM25 results, marked the dense request as a fallback, and left liveness at 200 and readiness at 503. Restart the test server after the drill.

## Shadow release and rollback

The optional release-state file is the local control-plane pointer. It records
the active, previous, and shadow model as immutable registry version/release-ID
pairs. Writes use a file lock, an atomic replacement, and an expected generation
so two operators cannot silently overwrite one another. Keep this runtime file
outside Git.

Initialize the pointer with the current champion, then stage a gate-approved
candidate. Each command prints the new generation.

```sh
make release RELEASE_ARGS='--state artifacts/release/state.json init --model-version 1'
make release RELEASE_ARGS='--state artifacts/release/state.json stage --model-version 2 --dense-timeout-ms 500 --expected-generation 1'
make serve SERVE_ARGS='--release-state artifacts/release/state.json'
```

For every dense search, the active worker produces the user-visible response.
The shadow worker runs as a separate process and task with its own timeout. Its
document IDs, latency, outcome, and lineage are logged under
`shadow_prediction`; its results can never replace the response. Prometheus
exports `scifact_shadow_requests_total` and
`scifact_shadow_duration_seconds`. BM25 and keyword-only requests are not
shadowed.

After checking shadow health and retrieval-quality evidence, promote the exact
staged release and restart the API to load the new pointer:

```sh
make release RELEASE_ARGS='--state artifacts/release/state.json promote --expected-generation 2'
make serve SERVE_ARGS='--release-state artifacts/release/state.json'
```

If the release is unhealthy, atomically swap back to the recorded previous
release and restart. Rollback reuses verified artifacts; it does not retrain or
rebuild them.

```sh
make release RELEASE_ARGS='--state artifacts/release/state.json rollback --expected-generation 3'
make serve SERVE_ARGS='--release-state artifacts/release/state.json'
```

The API verifies at startup that every pointer still resolves to the recorded
release ID and that active and shadow releases use the same prepared snapshot.
Promotion and rollback change only the control-plane pointer; an external
process supervisor or cloud deployment must perform the restart.

A local drill on November 23, 2026 used registered model 3 as active and model
4 as shadow. The active response returned five model-3 results while the
separate shadow worker logged five model-4 document IDs and exported one
successful shadow request at 5.266 ms. Promotion changed the active pointer to
model 4 at generation 3; rollback restored model 3 at generation 4. This
was followed by an API restart that loaded the original model-3 release with no
shadow. This verifies the manual local control path, not a cloud rollout or an
automated supervisor-driven restart.
