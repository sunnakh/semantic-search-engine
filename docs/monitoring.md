# Search service monitoring

This local setup uses Prometheus and Grafana on the same host as the API. The dashboard covers API scrape health, dense-worker readiness, request outcomes, fallback share, and p50 and p95 search latency. Prometheus evaluates one alert for dense-worker failure. Check its state on the Prometheus Alerts page.

## Start monitoring

Start a pinned release using the [serving guide](serving.md). Prometheus scrapes `127.0.0.1:8000` by default. For an API on another port, copy `prometheus.yml` and `alerts.yml` into a temporary directory, set the target in the copied `prometheus.yml`, and start Prometheus with that config.

From the project root, run:

```sh
promtool check config infra/monitoring/prometheus.yml
promtool test rules infra/monitoring/alerts.test.yml
prometheus --config.file=infra/monitoring/prometheus.yml \
  --web.listen-address=127.0.0.1:9090 \
  --storage.tsdb.path="$(mktemp -d)"
```

Open [Prometheus Targets](http://127.0.0.1:9090/targets) and confirm that `scifact-search` is up. The rules test exercises the hold period, recovery, and scrape failure. [Prometheus rule tests](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/)

In a separate Bash or Zsh terminal in the project root, start Grafana and set the local admin password at the prompt. The `--homepath` below uses a Homebrew installation. Point it to your Grafana installation directory on other systems.

```sh
grafana_state=$(mktemp -d)
printf "Grafana admin password: "
read -rs SCIFACT_GRAFANA_PASSWORD
printf "\n"
GF_SECURITY_ADMIN_PASSWORD="$SCIFACT_GRAFANA_PASSWORD" \
GF_PATHS_DATA="$grafana_state/data" \
GF_PATHS_LOGS="$grafana_state/log" \
GF_PATHS_PLUGINS="$grafana_state/plugins" \
GF_PATHS_PROVISIONING="$PWD/infra/monitoring/grafana/provisioning" \
GF_SERVER_HTTP_ADDR=127.0.0.1 \
SCIFACT_GRAFANA_DASHBOARDS="$PWD/infra/monitoring/grafana/dashboards" \
SCIFACT_PROMETHEUS_URL=http://127.0.0.1:9090 \
grafana server --homepath "$(brew --prefix grafana)/share/grafana"
```

Sign in as `admin` with the password you entered. Open the [SciFact service health dashboard](http://127.0.0.1:3000/d/scifact-service-health) and run search requests to populate the latency and request panels. Prometheus scrapes every five seconds. Rate panels need multiple scrapes and the fallback share needs recent dense traffic. In this setup the request counter covers validated searches and their fallback outcomes. Use API logs to investigate HTTP errors.

## Dense worker unavailable

`SciFactDenseWorkerUnavailable` fires when Prometheus can still scrape the API and observes `scifact_dense_ready = 0` for 15 seconds. Readiness returns HTTP 503, liveness remains 200, and dense searches can fall back to BM25. A failed scrape shows as `API scrape = 0` and warrants checking the API process, host, and configured target first.

1. Check [Prometheus Alerts](http://127.0.0.1:9090/alerts), `/health/ready`, and the API logs for worker exit or deadline errors. Prediction logs identify the release and indicate whether a query fell back.
2. Restart the API with the same pinned model version and manifest. Check `/api/v1/models` for the expected release, `/health/ready` for HTTP 200, and a dense search for `method: dense` and `fallback: false`.
3. If the issue started after a release change and recurs after restart, restart on the last known-good approved version with its matching manifest. Verify the expected model and data versions in `/api/v1/models`, repeat the dense search, and confirm that the alert clears after the worker readiness gauge returns to 1.

### Local failure drill

Use a separate test API process and point Prometheus at its port. Set `SCIFACT_API_URL` to the test API base URL. Wait for both health panels to show 1. Set `API_PID` to the test server process ID printed at startup. Check the command printed for its worker before sending the signal.

```sh
worker_pid=$(pgrep -P "$API_PID" -f 'src.serving.worker')
ps -p "$worker_pid" -o pid,ppid,command
```

After confirming the process belongs to the test API, run:

```sh
kill -KILL "$worker_pid"
curl -sS -w '\nHTTP %{http_code}\n' "$SCIFACT_API_URL/health/ready"
curl -fsS "$SCIFACT_API_URL/api/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"exercise and heart health","method":"dense","top_k":5}'
```

Expect HTTP 503 from readiness and a successful search response with `method: bm25` and `fallback: true`. Check that the alert changes to firing, recover with the pinned restart above, and verify that it clears. Stop the test API, Prometheus, and Grafana when finished.

## Monitor retrieval quality

Start the API and Prometheus using the setup above. Pick the prepared manifest used by the API and the `candidate-validation.json` report from a known-good, gate-approved MiniLM bundle for that snapshot. The candidate's training output contains this report beside its bundle manifest.

```sh
SCIFACT_MANIFEST=data/processed/scifact/YOUR_DATASET_VERSION/manifest.json
SCIFACT_REFERENCE=artifacts/models/candidates/YOUR_CANDIDATE_ID/candidate-validation.json
make monitor-quality MONITOR_ARGS="--manifest $SCIFACT_MANIFEST --reference-report $SCIFACT_REFERENCE"
```

The job runs immediately, then waits five minutes between replays. Set `--interval-seconds` to shorten the wait for a local drill. If your API uses a different port, pass its base URL as `--api-url`. The exporter listens on `127.0.0.1:9101`, which matches the `scifact-sentinel` scrape target.

Each replay asks the API for ten results per claim on the fixed, labeled validation set. SciFact has 162 claims on the current snapshot. The replay verifies the corpus and release lineage and computes nDCG@10 and Recall@5 from the rankings returned to callers, including any BM25 fallback. The startup check recomputes the reference scores from their saved rankings. The alert floors are the verified reference scores minus 0.02, with a minimum of zero. This is an absolute two-point tolerance for the runtime alert.

Grafana plots the two scores, their floors, fallback share, replay success, and time since the last successful replay. A replay that encounters an error publishes a failed status and withdraws its scores. Use the freshness panel and exporter logs to diagnose missing samples. The replay uses validation labels, so its results measure runtime retrieval on a fixed benchmark. Its search calls appear in API metrics and prediction logs.

## Sentinel quality low

`SciFactSentinelQualityLow` fires per metric when a complete replay stays below its floor for 15 seconds of observation. A single replay can hold that value until the next replay finishes.

1. Check the measured scores, fallback share, worker health, and time since the last successful replay. If the worker is degraded, use the [dense worker runbook](#dense-worker-unavailable).
2. Use the exporter log to confirm the release ID and dataset version. Compare the served release to the approved reference and inspect which recent release or configuration changed.
3. If a regression follows a release and persists with a healthy worker, restore the last known-good approved release with its matching prepared manifest. Confirm its lineage in `/api/v1/models`, wait for a complete replay, and check that both scores recover above their floors and the alert clears.

### Local verification

The live replay of MiniLM registry version 3 matched its saved validation report across all 162 claims, with nDCG@10 of 0.7017 and Recall@5 of 0.7716. In a simulated regression, a test proxy returned empty rankings. Both scores fell to zero and their alerts fired. Restoring the original rankings recovered both scores and cleared the alerts.
