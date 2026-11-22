.PHONY: ingest prepare baseline baselines dense-baseline
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
