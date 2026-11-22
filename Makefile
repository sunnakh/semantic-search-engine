.PHONY: ingest prepare

ingest:
	uv run --locked python -m src.data.ingest

prepare:
	uv run --locked python -m src.data.prepare
