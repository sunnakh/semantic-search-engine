.PHONY: ingest prepare baseline baselines dense-baseline train-candidate pipeline serve
ingest:
	uv run --locked python -m src.data.ingest

prepare:
	uv run --locked python -m src.data.prepare

baseline:
	uv run --locked python -m src.evaluation.benchmark

baselines:
	uv run --locked python -m src.evaluation.benchmark --method all

dense-baseline:
	uv run --locked --extra dense python -m src.evaluation.benchmark --method dense

train-candidate:
	uv run --locked --extra dense python -m src.training.candidate

pipeline:
	MLFLOW_DISABLE_TELEMETRY=true uv run --locked --extra dense --extra registry python -m src.training.pipeline

serve:
	MLFLOW_DISABLE_TELEMETRY=true uv run --locked --extra dense --extra registry --extra serving python -m src.serving.app $(SERVE_ARGS)
