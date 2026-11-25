.PHONY: ingest prepare baseline baselines dense-baseline train-candidate pipeline serve release export-release ui monitor-quality train-qwen-embedding evaluate-qwen-embedding train-qwen evaluate-qwen
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

train-qwen:
	uv run --locked --isolated --extra qwen-mlx python -m src.training.qwen_mlx train $(QWEN_ARGS)

evaluate-qwen:
	uv run --locked --isolated --extra qwen-mlx python -m src.training.qwen_mlx evaluate $(QWEN_ARGS)

train-qwen-embedding:
	uv run --locked --isolated --extra dense --extra qwen python -m src.training.qwen_lora train $(QWEN_ARGS)

evaluate-qwen-embedding:
	uv run --locked --isolated --extra dense --extra qwen python -m src.training.qwen_lora evaluate $(QWEN_ARGS)

pipeline:
	MLFLOW_DISABLE_TELEMETRY=true uv run --locked --extra dense --extra registry python -m src.training.pipeline

serve:
	MLFLOW_DISABLE_TELEMETRY=true uv run --locked --extra dense --extra registry --extra serving python -m src.serving.app $(SERVE_ARGS)

release:
	MLFLOW_DISABLE_TELEMETRY=true uv run --locked --extra dense --extra registry python -m src.serving.control $(RELEASE_ARGS)

export-release:
	MLFLOW_DISABLE_TELEMETRY=true uv run --locked --extra dense --extra registry python -m src.serving.portable $(EXPORT_ARGS)

ui:
	uv run --locked --isolated --extra ui python -m streamlit run src/ui/app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false $(UI_ARGS)

monitor-quality:
	uv run --locked --isolated --extra dense --extra serving python -m src.monitoring.sentinel $(MONITOR_ARGS)
