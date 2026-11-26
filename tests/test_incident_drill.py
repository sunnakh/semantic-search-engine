import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from urllib.request import Request, urlopen

DRILL_AVAILABLE = all(find_spec(name) for name in ("mlflow", "numpy", "fastapi"))
if DRILL_AVAILABLE:
    from src.monitoring.drill import alert_matches, quality_recovered
    from src.monitoring.fault_proxy import fault_proxy, inject_empty_ranking


@unittest.skipUnless(DRILL_AVAILABLE, "dense, registry, and serving extras required")
class IncidentDrillTests(unittest.TestCase):
    def test_fault_only_changes_dense_rankings_for_the_selected_release(self) -> None:
        original = {"release_id": "candidate", "method": "dense", "results": ["paper"]}
        self.assertEqual(inject_empty_ranking(original, "candidate")["results"], [])
        self.assertEqual(original["results"], ["paper"])
        self.assertEqual(inject_empty_ranking(original, "other"), original)
        lexical = {**original, "method": "bm25"}
        self.assertEqual(inject_empty_ranking(lexical, "candidate"), lexical)

    def test_alert_requires_both_metrics_and_clears_only_when_all_are_absent(
        self,
    ) -> None:
        def body(alerts):
            return json.dumps({"data": {"alerts": alerts}}).encode()

        alerts = [
            {
                "labels": {"alertname": "SciFactSentinelQualityLow", "metric": metric},
                "state": "firing",
            }
            for metric in ("ndcg_at_10", "recall_at_5")
        ]
        self.assertFalse(alert_matches(body([]), True))
        self.assertFalse(alert_matches(body(alerts[:1]), True))
        self.assertTrue(alert_matches(body(alerts), True))
        alerts[0]["state"] = "pending"
        self.assertFalse(alert_matches(body(alerts), True))
        self.assertFalse(alert_matches(body(alerts), False))
        self.assertTrue(alert_matches(body([]), False))
        self.assertFalse(
            quality_recovered(json.dumps({"data": {"result": []}}).encode())
        )

    def test_proxy_preserves_metadata_and_credentials_and_recovery_needs_new_release(
        self,
    ) -> None:
        current = {
            "release_id": "candidate",
            "method": "dense",
            "results": [{"document_id": "paper"}],
        }
        markers = []

        class Upstream(BaseHTTPRequestHandler):
            def do_GET(self):
                self.respond({"release_id": current["release_id"]})

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                markers.append(self.headers.get("X-SciFact-Sentinel"))
                self.respond(current)

            def respond(self, payload):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with fault_proxy(
                f"http://127.0.0.1:{server.server_port}", "candidate", 0
            ) as url:
                with urlopen(url + "/api/v1/models", timeout=3) as response:
                    self.assertEqual(json.load(response), {"release_id": "candidate"})
                request = Request(
                    url + "/api/v1/search",
                    data=b'{"query":"science"}',
                    headers={
                        "Content-Type": "application/json",
                        "X-SciFact-Sentinel": "token",
                    },
                )
                with urlopen(request, timeout=3) as response:
                    self.assertEqual(json.load(response)["results"], [])
                current["release_id"] = "previous"
                with urlopen(request, timeout=3) as response:
                    recovered = json.load(response)
                    self.assertEqual(recovered["results"], current["results"])
                    self.assertEqual(recovered["release_id"], "previous")
                self.assertEqual(markers, ["token", "token"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
