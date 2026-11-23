FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv
FROM python:3.13.15-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS python-base
FROM python-base AS dependencies

WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-python-downloads \
    --extra dense --extra registry --extra serving

FROM python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY --from=dependencies /app/.venv /app/.venv
COPY src/ src/
RUN useradd --create-home --uid 10001 scifact
USER scifact

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/ready', timeout=2).close()"]

ENTRYPOINT ["/app/.venv/bin/python", "-m", "src.serving.app", "--host", "0.0.0.0"]
