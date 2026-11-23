import asyncio
import json
import sys
import unittest
from contextlib import AsyncExitStack
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from src.data.prepared import PreparedSplit
from src.documents import Document
from src.retrieval.keyword import KeywordIndex
from src.retrieval.lexical import BM25Index

SERVING_AVAILABLE = all(
    find_spec(name) for name in ("fastapi", "mlflow", "numpy", "prometheus_client")
)
if SERVING_AVAILABLE:
    from starlette.types import Message, Scope

    from src.monitoring.drift import reference_counts
    from src.serving.app import MAX_BODY_BYTES, create_app
    from src.serving.release import LoadedRelease, Release
    from src.serving.worker import DenseUnavailable, DenseWorker, Hit


class FixtureWorker:
    ready = True

    async def search(self, query: str, top_k: int) -> list["Hit"]:
        if not self.ready:
            raise DenseUnavailable("dense worker is offline")
        return [Hit("d2", 0.8)]

    async def close(self) -> None:
        pass


@unittest.skipUnless(SERVING_AVAILABLE, "dense, registry, and serving extras required")
class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        root = Path("unused")
        documents = [
            Document("d1", "Heart health", "Exercise improves heart health."),
            Document("d2", "Exercise", "Exercise supports sleep."),
        ]
        metadata = Release(
            release_id="release",
            model_version="1",
            candidate_id="candidate",
            training_commit="a" * 40,
            dataset_version="dataset",
            prepared_manifest_sha256="prepared-hash",
            bundle_manifest_sha256="bundle-hash",
            keyword_version="keyword-index",
            bm25_version="bm25-index",
            dense_timeout_ms=300,
        )
        release = LoadedRelease(
            metadata,
            root,
            {item.document_id: item for item in documents},
            KeywordIndex(documents),
            BM25Index(documents),
        )
        self.dense = FixtureWorker()
        self.training = PreparedSplit(
            "dataset",
            "prepared-hash",
            documents,
            {"q1": "  a  ", "q2": "x" * 32, "q3": "x" * 33, "excluded": "x" * 512},
            {},
            ("excluded",),
        )
        self.app = create_app(root, root, 1, drift_token="test-token")
        stack = AsyncExitStack()
        self.addAsyncCleanup(stack.aclose)
        with (
            patch("src.serving.app.load_release", return_value=release),
            patch("src.serving.app.load_split", return_value=self.training),
            patch.object(DenseWorker, "start", return_value=self.dense),
        ):
            await stack.enter_async_context(self.app.router.lifespan_context(self.app))

    async def request(
        self,
        path: str,
        body: bytes | None = None,
        *,
        sentinel_token: bytes | None = None,
    ) -> tuple[int, bytes]:
        headers = [(b"content-type", b"application/json")]
        if sentinel_token is not None:
            headers.append((b"x-scifact-sentinel", sentinel_token))
        scope: Scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET" if body is None else "POST",
            "path": path,
            "headers": headers,
            "query_string": b"",
        }
        messages: list[Message] = []

        async def receive() -> "Message":
            return {"type": "http.request", "body": body or b"", "more_body": False}

        async def send(message: "Message") -> None:
            messages.append(message)

        await self.app(scope, receive, send)
        return messages[0]["status"], b"".join(
            message.get("body", b"") for message in messages[1:]
        )

    async def test_drift_counts_regular_traffic_and_excludes_sentinel(self) -> None:
        self.assertEqual(reference_counts(self.training), [2, 1, 0, 0, 0])
        search_path = "/api/v1/search"
        payload = b'{"query":"  Exercise  ","top_k":1}'
        self.assertEqual((await self.request(search_path, payload))[0], 200)
        self.assertEqual(
            (await self.request(search_path, payload, sentinel_token=b"test-token"))[0],
            200,
        )
        self.assertEqual(
            (await self.request(search_path, payload, sentinel_token=b"wrong"))[0],
            403,
        )
        _, metrics = await self.request("/metrics")
        self.assertIn(
            b'scifact_query_length_requests_total{bucket="1-32",release_id="release"} 1.0\n',
            metrics,
        )

    async def test_dense_returns_scores_documents_and_lineage(self) -> None:
        status, body = await self.request(
            "/api/v1/search", b'{"query":"Exercise","method":"dense","top_k":1}'
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["method"], "dense")
        self.assertEqual(result["model_version"], "1")
        self.assertEqual(result["results"][0]["score"], 0.8)
        self.assertEqual(result["results"][0]["document_id"], "d2")
        _, metadata = await self.request("/api/v1/models")
        self.assertEqual(json.loads(metadata)["release_id"], result["release_id"])
        self.assertEqual(json.loads(metadata)["training_commit"], "a" * 40)
        self.assertEqual((await self.request("/api/v1/documents/missing"))[0], 404)
        self.assertEqual((await self.request("/health/ready"))[0], 200)

    async def test_degraded_worker_uses_bm25_and_records_fallback(self) -> None:
        counter = b'scifact_search_requests_total{method="dense",outcome="fallback"}'
        self.assertIn(counter + b" 0.0", (await self.request("/metrics"))[1])
        self.dense.ready = False
        with self.assertLogs("scifact.predictions", level="INFO") as logs:
            status, body = await self.request(
                "/api/v1/search", b'{"query":"Heart health","top_k":1}'
            )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["method"], "bm25")
        self.assertEqual(result["model_version"], "bm25-index")
        self.assertEqual(result["results"][0]["document_id"], "d1")
        self.assertNotIn("Heart health", logs.output[0])
        self.assertIn('"query_hash"', logs.output[0])
        self.assertEqual((await self.request("/health/live"))[0], 200)
        self.assertEqual((await self.request("/health/ready"))[0], 503)
        self.assertIn(counter + b" 1.0", (await self.request("/metrics"))[1])

    async def test_validation_rejects_bad_and_oversized_requests(self) -> None:
        for payload in (
            {"query": " "},
            {"query": "x" * 513},
            {"query": "private", "top_k": True},
            {"query": "private", "top_k": 0},
            {"query": "private", "method": "unknown"},
            {"query": "private", "secret-extra": True},
        ):
            with self.subTest(payload=payload):
                status, body = await self.request(
                    "/api/v1/search", json.dumps(payload).encode()
                )
                self.assertEqual(status, 422)
                self.assertNotIn(b"private", body)
                self.assertNotIn(b"secret-extra", body)
        status, _ = await self.request("/api/v1/search", b"x" * (MAX_BODY_BYTES + 1))
        self.assertEqual(status, 413)


@unittest.skipUnless(SERVING_AVAILABLE, "serving extras required")
class DenseWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_flight_timeout_discards_worker(self) -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys, time; sys.stdin.readline(); time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        worker = DenseWorker(process, timeout_ms=30)
        self.addAsyncCleanup(worker.close)
        with self.assertRaisesRegex(DenseUnavailable, "deadline exceeded"):
            await worker.search("test", top_k=1)
        self.assertFalse(worker.ready)
        with self.assertRaisesRegex(DenseUnavailable, "unavailable"):
            await worker.search("test again", top_k=1)
