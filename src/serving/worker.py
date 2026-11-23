"""Isolate dense inference behind a bounded local subprocess connection."""

import asyncio
import json
import sys
from contextlib import redirect_stdout, suppress
from pathlib import Path
from typing import NamedTuple


class Hit(NamedTuple):
    document_id: str
    score: float


class DenseUnavailable(RuntimeError):
    """The dense worker cannot accept or complete this search."""


class DenseWorker:
    def __init__(self, process: asyncio.subprocess.Process, timeout_ms: int) -> None:
        self.process = process
        self._timeout = timeout_ms / 1000
        self._lock = asyncio.Lock()
        self._stopped = False

    @classmethod
    async def start(cls, bundle: Path, timeout_ms: int) -> "DenseWorker":
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.serving.worker",
            str(bundle),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        worker = cls(process, timeout_ms)
        try:
            assert process.stdout is not None
            if await asyncio.wait_for(process.stdout.readline(), 60) != b"ready\n":
                raise DenseUnavailable("dense worker failed to load the model")
        except BaseException:
            await worker.close()
            raise
        return worker

    @property
    def ready(self) -> bool:
        return not self._stopped and self.process.returncode is None

    def _stop(self) -> None:
        self._stopped = True
        if self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.kill()

    async def close(self) -> None:
        self._stop()
        await self.process.wait()

    async def search(self, query: str, top_k: int) -> list[Hit]:
        acquired = False
        try:
            async with asyncio.timeout(self._timeout):
                await self._lock.acquire()
                acquired = True
                if not self.ready:
                    raise DenseUnavailable("dense worker is unavailable")
                request = json.dumps([query, top_k]).encode() + b"\n"
                assert (
                    self.process.stdin is not None and self.process.stdout is not None
                )
                self.process.stdin.write(request)
                await self.process.stdin.drain()
                response = await self.process.stdout.readline()
                if not response:
                    raise EOFError("dense worker closed its response stream")
                return [Hit(*row) for row in json.loads(response)]
        except TimeoutError as error:
            if acquired:
                self._stop()
            raise DenseUnavailable("dense deadline exceeded") from error
        except (EOFError, OSError, ValueError, TypeError) as error:
            self._stop()
            raise DenseUnavailable("dense worker failed") from error
        except asyncio.CancelledError:
            if acquired:
                self._stop()
            raise
        finally:
            if acquired:
                self._lock.release()


def main() -> int:
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        import torch

        from ..retrieval.dense import DenseIndex, MiniLMEncoder, load_minilm

        torch.set_num_threads(1)
        bundle = Path(sys.argv[1])
        index = DenseIndex.load(
            bundle / "index", MiniLMEncoder(load_minilm(bundle / "encoder"))
        )
        index.search("scientific evidence", 1)
        print("ready", file=output, flush=True)
        for line in sys.stdin:
            try:
                query, top_k = json.loads(line)
                hits = [
                    (hit.document_id, hit.score) for hit in index.search(query, top_k)
                ]
                print(json.dumps(hits, allow_nan=False), file=output, flush=True)
            except (
                IndexError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                print(f"Dense worker failed: {type(error).__name__}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
