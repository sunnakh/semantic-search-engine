"""Serve a pinned SciFact release through FastAPI."""

import argparse
import asyncio
import hmac
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import get_args
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..data.prepared import DEFAULT_PROCESSED_ROOT, load_split
from ..monitoring.drift import QueryLengthMetrics
from ..registry.tracking import TRACKING_ROOT
from .control import ReleaseState, ReleaseStore
from .portable import SNAPSHOT_MANIFEST, load_portable_release
from .release import DEFAULT_DENSE_TIMEOUT_MS, LoadedRelease, load_release
from .schemas import (
    SEARCH_PATH,
    SENTINEL_HEADER,
    DocumentResponse,
    Method,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .worker import DenseUnavailable, DenseWorker, Hit

MAX_BODY_BYTES = 8192
PREDICTIONS = logging.getLogger("scifact.predictions")


class LimitSearchBody:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"] != SEARCH_PATH
            or scope["method"] != "POST"
        ):
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                await JSONResponse({"detail": "search request is too large"}, 413)(
                    scope, receive, send
                )
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_body() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_body, send)


@dataclass
class Runtime:
    release: LoadedRelease
    dense: DenseWorker
    release_state: ReleaseState | None = None
    shadow_release: LoadedRelease | None = None
    shadow_dense: DenseWorker | None = None
    shadow_tasks: set[asyncio.Task[None]] | None = None

    async def retrieve(self, payload: SearchRequest) -> tuple[Method, list[Hit]]:
        method = payload.method
        if method == "dense":
            try:
                return method, await self.dense.search(payload.query, payload.top_k)
            except DenseUnavailable:
                method = "bm25"
        index = self.release.bm25 if method == "bm25" else self.release.keyword
        hits = await asyncio.to_thread(
            lambda: index.search(payload.query, payload.top_k)
        )
        return method, [Hit(hit.document_id, float(hit.score)) for hit in hits]


def create_app(
    prepared_manifest: Path | None,
    tracking_root: Path,
    model_version: int | None,
    dense_timeout_ms: int = DEFAULT_DENSE_TIMEOUT_MS,
    *,
    drift_token: str | None = None,
    release_state_path: Path | None = None,
    portable_root: Path | None = None,
) -> FastAPI:
    if (
        sum(
            source is not None
            for source in (model_version, release_state_path, portable_root)
        )
        != 1
    ):
        raise ValueError(
            "select exactly one model version, release state, or portable release"
        )
    if (model_version is not None and model_version < 1) or dense_timeout_ms < 1:
        raise ValueError("model versions and timeouts must be positive")
    if portable_root is not None:
        if prepared_manifest is not None:
            raise ValueError("a portable release contains its own prepared manifest")
        prepared_manifest = portable_root / SNAPSHOT_MANIFEST
    if prepared_manifest is None:
        raise ValueError("a prepared manifest is required")
    data_manifest = prepared_manifest

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with TemporaryDirectory(prefix="scifact-release-") as directory:
            state = (
                ReleaseStore(release_state_path).read() if release_state_path else None
            )
            active_version = state.active.model_version if state else model_version
            active_timeout = (
                state.active.dense_timeout_ms if state else dense_timeout_ms
            )
            if portable_root is not None:
                release = await asyncio.to_thread(load_portable_release, portable_root)
            else:
                assert active_version is not None
                release = await asyncio.to_thread(
                    load_release,
                    data_manifest,
                    tracking_root,
                    active_version,
                    active_timeout,
                    Path(directory) / "active",
                )
            if state and release.metadata.release_id != state.active.release_id:
                raise ValueError("active release pointer does not match the registry")
            shadow_release = None
            if state and state.shadow:
                shadow_release = await asyncio.to_thread(
                    load_release,
                    data_manifest,
                    tracking_root,
                    state.shadow.model_version,
                    state.shadow.dense_timeout_ms,
                    Path(directory) / "shadow",
                )
                if shadow_release.metadata.release_id != state.shadow.release_id:
                    raise ValueError(
                        "shadow release pointer does not match the registry"
                    )
                if (
                    shadow_release.metadata.dataset_version
                    != release.metadata.dataset_version
                    or shadow_release.metadata.prepared_manifest_sha256
                    != release.metadata.prepared_manifest_sha256
                ):
                    raise ValueError("active and shadow releases must use one snapshot")
            if drift is not None:
                training = await asyncio.to_thread(load_split, data_manifest, "train")
                if (
                    training.dataset_version != release.metadata.dataset_version
                    or training.manifest_sha256
                    != release.metadata.prepared_manifest_sha256
                ):
                    raise ValueError(
                        "the drift reference must match the served snapshot"
                    )
                drift.configure(training, release.metadata.release_id)
            dense = await DenseWorker.start(
                release.bundle, release.metadata.dense_timeout_ms
            )
            shadow_dense = (
                await DenseWorker.start(
                    shadow_release.bundle, shadow_release.metadata.dense_timeout_ms
                )
                if shadow_release
                else None
            )
            app.state.runtime = Runtime(
                release,
                dense,
                state,
                shadow_release,
                shadow_dense,
                set(),
            )
            try:
                yield
            finally:
                runtime: Runtime = app.state.runtime
                tasks = runtime.shadow_tasks or set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if shadow_dense:
                    await shadow_dense.close()
                await dense.close()

    app = FastAPI(title="SciFact Search", lifespan=lifespan)
    app.add_middleware(LimitSearchBody)
    query_key = os.environ.get("SCIFACT_QUERY_HASH_KEY")
    hash_key = query_key.encode() if query_key else secrets.token_bytes(32)
    metrics = CollectorRegistry()
    drift = QueryLengthMetrics(metrics) if drift_token is not None else None
    searches = Counter(
        "scifact_search_requests",
        "Validated search requests",
        ("method", "outcome"),
        registry=metrics,
    )
    latency = Histogram(
        "scifact_search_duration_seconds",
        "Search handling time including retrieval and result assembly",
        ("method",),
        registry=metrics,
    )
    # Publish zero samples so Prometheus can measure the first request.
    for method in get_args(Method):
        searches.labels(method=method, outcome="success")
        latency.labels(method=method)
    searches.labels(method="dense", outcome="fallback")
    worker_health = Gauge(
        "scifact_dense_ready", "Dense worker readiness", registry=metrics
    )
    worker_health.set_function(lambda: int(app.state.runtime.dense.ready))
    shadow_searches = Counter(
        "scifact_shadow_requests",
        "Shadow retrieval attempts",
        ("outcome", "release_id"),
        registry=metrics,
    )
    shadow_latency = Histogram(
        "scifact_shadow_duration_seconds",
        "Shadow retrieval latency outside the response path",
        ("release_id",),
        registry=metrics,
    )

    async def run_shadow(
        runtime: Runtime, payload: SearchRequest, request_id: str, query_hash: str
    ) -> None:
        shadow = runtime.shadow_release
        worker = runtime.shadow_dense
        if shadow is None or worker is None:
            return
        start = perf_counter()
        outcome = "success"
        document_ids: list[str] = []
        try:
            hits = await worker.search(payload.query, payload.top_k)
            document_ids = [hit.document_id for hit in hits]
        except DenseUnavailable:
            outcome = "unavailable"
        duration = perf_counter() - start
        release_id = shadow.metadata.release_id
        shadow_searches.labels(outcome=outcome, release_id=release_id).inc()
        shadow_latency.labels(release_id=release_id).observe(duration)
        PREDICTIONS.info(
            "%s",
            json.dumps(
                {
                    "event": "shadow_prediction",
                    "request_id": request_id,
                    "release_id": release_id,
                    "model_version": shadow.metadata.model_version,
                    "query_hash": query_hash,
                    "document_ids": document_ids,
                    "latency_ms": round(duration * 1000, 3),
                    "outcome": outcome,
                }
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        errors = [{"type": item["type"]} for item in error.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        runtime: Runtime = app.state.runtime
        if not runtime.dense.ready:
            return JSONResponse({"status": "degraded"}, status_code=503)
        return JSONResponse(
            {"status": "ready", "release_id": runtime.release.metadata.release_id}
        )

    @app.get("/api/v1/models")
    async def models() -> dict[str, object]:
        runtime: Runtime = app.state.runtime
        return {
            **asdict(runtime.release.metadata),
            "methods": {"dense": runtime.dense.ready, "bm25": True, "keyword": True},
            "release_state": (
                asdict(runtime.release_state) if runtime.release_state else None
            ),
        }

    @app.get("/api/v1/documents/{document_id}")
    async def document(document_id: str) -> DocumentResponse:
        runtime: Runtime = app.state.runtime
        result = runtime.release.documents.get(document_id)
        if result is None:
            raise HTTPException(status_code=404, detail="document not found")
        return DocumentResponse.from_document(result)

    @app.get("/metrics", include_in_schema=False)
    async def prometheus() -> Response:
        return Response(generate_latest(metrics), media_type=CONTENT_TYPE_LATEST)

    @app.post(SEARCH_PATH)
    async def search(payload: SearchRequest, request: Request) -> SearchResponse:
        marker = request.headers.get(SENTINEL_HEADER)
        if (
            drift_token is not None
            and marker is not None
            and not hmac.compare_digest(drift_token.encode(), marker.encode())
        ):
            raise HTTPException(status_code=403, detail="invalid sentinel token")
        start = perf_counter()
        runtime: Runtime = app.state.runtime
        release = runtime.release
        metadata = release.metadata
        request_id = uuid4().hex
        query_hash = hmac.digest(hash_key, payload.query.encode(), "sha256").hex()
        effective_method, hits = await runtime.retrieve(payload)
        fallback = effective_method != payload.method

        model_versions = {
            "dense": metadata.model_version,
            "bm25": metadata.bm25_version,
            "keyword": metadata.keyword_version,
        }
        version = model_versions[effective_method]
        results = []
        for doc_id, score in hits:
            document = release.documents[doc_id]
            results.append(
                SearchHit(
                    document_id=doc_id,
                    title=document.title,
                    abstract=document.text,
                    score=score,
                )
            )
        duration = perf_counter() - start
        latency.labels(method=payload.method).observe(duration)
        searches.labels(
            method=payload.method, outcome="fallback" if fallback else "success"
        ).inc()
        if drift is not None and marker is None:
            drift.observe(len(payload.query))
        latency_ms = round(duration * 1000, 3)
        PREDICTIONS.info(
            "%s",
            json.dumps(
                {
                    "event": "search_prediction",
                    "request_id": request_id,
                    "release_id": metadata.release_id,
                    "requested_method": payload.method,
                    "method": effective_method,
                    "model_version": version,
                    "query_hash": query_hash,
                    "query_length": len(payload.query),
                    "document_ids": [result.document_id for result in results],
                    "latency_ms": latency_ms,
                    "fallback": fallback,
                }
            ),
        )
        if payload.method == "dense" and runtime.shadow_release is not None:
            task = asyncio.create_task(
                run_shadow(runtime, payload, request_id, query_hash),
                name=f"shadow-{request_id}",
            )
            assert runtime.shadow_tasks is not None
            runtime.shadow_tasks.add(task)
            task.add_done_callback(runtime.shadow_tasks.discard)
        return SearchResponse(
            method=effective_method,
            model_version=version,
            release_id=metadata.release_id,
            request_id=request_id,
            fallback=fallback,
            results=results,
            latency_ms=latency_ms,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="prepared SciFact manifest")
    parser.add_argument("--tracking-root", type=Path)
    release = parser.add_mutually_exclusive_group(required=True)
    release.add_argument("--model-version", type=int)
    release.add_argument("--release-state", type=Path)
    release.add_argument(
        "--portable-release", type=Path, help="exported release directory"
    )
    parser.add_argument("--dense-timeout-ms", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--monitor-drift", action="store_true")
    args = parser.parse_args()
    drift_token = (
        os.environ.get("SCIFACT_SENTINEL_TOKEN") if args.monitor_drift else None
    )
    if args.monitor_drift and len(drift_token or "") < 32:
        parser.error(
            "set SCIFACT_SENTINEL_TOKEN to a random token of at least 32 characters"
        )
    manifest = args.manifest
    if args.portable_release is not None:
        if any(
            value is not None
            for value in (manifest, args.tracking_root, args.dense_timeout_ms)
        ):
            parser.error(
                "a portable release supplies its own manifest and timeout and does not use a registry"
            )
    elif manifest is None:
        manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
        if len(manifests) != 1:
            parser.error(
                "run make prepare or select exactly one snapshot with --manifest"
            )
        (manifest,) = manifests

    PREDICTIONS.addHandler(logging.StreamHandler())
    PREDICTIONS.setLevel(logging.INFO)
    PREDICTIONS.propagate = False
    uvicorn.run(
        create_app(
            manifest,
            args.tracking_root if args.tracking_root is not None else TRACKING_ROOT,
            args.model_version,
            args.dense_timeout_ms
            if args.dense_timeout_ms is not None
            else DEFAULT_DENSE_TIMEOUT_MS,
            drift_token=drift_token,
            release_state_path=args.release_state,
            portable_root=args.portable_release,
        ),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
