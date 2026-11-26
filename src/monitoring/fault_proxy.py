"""Local fault injection for the simulated SciFact incident drill."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ..serving.schemas import SEARCH_PATH, SENTINEL_HEADER


def inject_empty_ranking(response: dict, release_id: str) -> dict:
    if response.get("release_id") == release_id and response.get("method") == "dense":
        return {**response, "results": []}
    return response


@contextmanager
def fault_proxy(api_url: str, release_id: str, port: int) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/api/v1/models":
                self.send_error(404)
                return
            self.forward(None)

        def do_POST(self) -> None:
            if self.path != SEARCH_PATH:
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400)
                return
            if not 0 < size <= 8192:
                self.send_error(413)
                return
            self.forward(self.rfile.read(size))

        def forward(self, body: bytes | None) -> None:
            headers = {"Content-Type": "application/json"}
            marker = self.headers.get(SENTINEL_HEADER)
            if marker is not None:
                headers[SENTINEL_HEADER] = marker
            request = Request(
                api_url.rstrip("/") + self.path, data=body, headers=headers
            )
            try:
                with urlopen(request, timeout=5) as upstream:
                    response = json.load(upstream)
                if not isinstance(response, dict):
                    raise ValueError("upstream response must be an object")
                if body is not None:
                    response = inject_empty_ranking(response, release_id)
                payload = json.dumps(response).encode()
            except HTTPError as error:
                self.send_error(error.code)
                error.close()
                return
            except (OSError, TypeError, ValueError):
                self.send_error(502)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
