"""Check warm retrieval latency with concurrent loopback HTTP clients."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.request import Request, urlopen

from ..data.prepared import load_validation


def search(url: str, query: str) -> tuple[float, float, bool]:
    request = Request(
        url,
        data=json.dumps({"query": query, "method": "dense", "top_k": 5}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = perf_counter()
    with urlopen(request, timeout=5) as response:
        result = json.load(response)
    return (
        (perf_counter() - start) * 1000,
        result["latency_ms"],
        (
            result["method"] == "dense"
            and not result["fallback"]
            and bool(result["results"])
        ),
    )


def p95(values: list[float]) -> float:
    return sorted(values)[ceil(0.95 * len(values)) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/search")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--requests", type=int, default=150)
    args = parser.parse_args()
    if args.users < 1 or args.requests < args.users:
        parser.error("requests must be at least the positive concurrent user count")

    validation = load_validation(args.manifest)
    queries = [validation.queries[key] for key in sorted(validation.queries)]
    for query in queries[-10:]:
        if not search(args.url, query)[2]:
            raise RuntimeError("dense retrieval failed during warmup")

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.users) as clients:
        futures = [
            clients.submit(search, args.url, queries[index % len(queries)])
            for index in range(args.requests)
        ]
        samples = []
        errors = 0
        for future in futures:
            try:
                samples.append(future.result())
            except (OSError, ValueError, KeyError, TypeError):
                errors += 1
    client_ms = [sample[0] for sample in samples]
    server_ms = [sample[1] for sample in samples]
    invalid = sum(not sample[2] for sample in samples)
    report = {
        "concurrent_users": args.users,
        "requests": args.requests,
        "warmup_requests": min(10, len(queries)),
        "http_errors": errors,
        "fallback_or_empty": invalid,
        "responses_per_second": round(len(samples) / (perf_counter() - started), 2),
        "client_p50_ms": round(median(client_ms), 3) if client_ms else None,
        "client_p95_ms": round(p95(client_ms), 3) if client_ms else None,
        "server_p50_ms": round(median(server_ms), 3) if server_ms else None,
        "server_p95_ms": round(p95(server_ms), 3) if server_ms else None,
        "target_server_p95_ms": 500,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(errors or invalid or not server_ms or p95(server_ms) > 500))


if __name__ == "__main__":
    raise SystemExit(main())
