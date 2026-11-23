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

from ..data.prepared import DEFAULT_PROCESSED_ROOT
from ..registry.tracking import TRACKING_ROOT
from .release import LoadedRelease, load_release
from .schemas import (
    SEARCH_PATH,
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


@dataclass(frozen=True)
class Runtime:
    release: LoadedRelease
    dense: DenseWorker

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
    prepared_manifest: Path,
    tracking_root: Path,
    model_version: int,
    dense_timeout_ms: int = 300,
) -> FastAPI:
    if model_version < 1 or dense_timeout_ms < 1:
        raise ValueError("model version and dense timeout must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with TemporaryDirectory(prefix="scifact-release-") as directory:
            release = await asyncio.to_thread(
                load_release,
                prepared_manifest,
                tracking_root,
                model_version,
                dense_timeout_ms,
                Path(directory),
            )
            dense = await DenseWorker.start(release.bundle, dense_timeout_ms)
            app.state.runtime = Runtime(release, dense)
            try:
                yield
            finally:
                await dense.close()

    app = FastAPI(title="SciFact Search", lifespan=lifespan)
    app.add_middleware(LimitSearchBody)
    query_key = os.environ.get("SCIFACT_QUERY_HASH_KEY")
    hash_key = query_key.encode() if query_key else secrets.token_bytes(32)
    metrics = CollectorRegistry()
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
    worker_health = Gauge(
        "scifact_dense_ready", "Dense worker readiness", registry=metrics
    )
    worker_health.set_function(lambda: int(app.state.runtime.dense.ready))

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
    async def search(payload: SearchRequest) -> SearchResponse:
        start = perf_counter()
        runtime: Runtime = app.state.runtime
        release = runtime.release
        metadata = release.metadata
        request_id = uuid4().hex
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
                    "query_hash": hmac.digest(
                        hash_key, payload.query.encode(), "sha256"
                    ).hex(),
                    "query_length": len(payload.query),
                    "document_ids": [result.document_id for result in results],
                    "latency_ms": latency_ms,
                    "fallback": fallback,
                }
            ),
        )
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
    parser.add_argument("--tracking-root", type=Path, default=TRACKING_ROOT)
    parser.add_argument("--model-version", type=int, required=True)
    parser.add_argument("--dense-timeout-ms", type=int, default=300)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    manifest = args.manifest
    if manifest is None:
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
            manifest, args.tracking_root, args.model_version, args.dense_timeout_ms
        ),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
