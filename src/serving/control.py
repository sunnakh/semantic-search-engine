"""Manage the active, previous, and shadow SciFact release pointers."""

import argparse
import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from ..data.prepared import DEFAULT_PROCESSED_ROOT
from ..registry.tracking import TRACKING_ROOT
from .release import load_release


@dataclass(frozen=True)
class ReleasePointer:
    model_version: int
    dense_timeout_ms: int
    release_id: str

    @classmethod
    def parse(cls, value: object) -> "ReleasePointer":
        if not isinstance(value, dict):
            raise TypeError("release pointer must be an object")
        version = value.get("model_version")
        timeout = value.get("dense_timeout_ms")
        release_id = value.get("release_id")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
            or not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or timeout < 1
            or not isinstance(release_id, str)
            or len(release_id) != 64
        ):
            raise ValueError("release pointer is invalid")
        return cls(version, timeout, release_id)


@dataclass(frozen=True)
class ReleaseState:
    generation: int
    active: ReleasePointer
    previous: ReleasePointer | None = None
    shadow: ReleasePointer | None = None
    schema_version: int = 1

    @classmethod
    def parse(cls, value: object) -> "ReleaseState":
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("unsupported release-state schema")
        generation = value.get("generation")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
        ):
            raise ValueError("release-state generation must be positive")

        def optional(name: str) -> ReleasePointer | None:
            item = value.get(name)
            return None if item is None else ReleasePointer.parse(item)

        state = cls(
            generation=generation,
            active=ReleasePointer.parse(value.get("active")),
            previous=optional("previous"),
            shadow=optional("shadow"),
        )
        pointers = [
            item for item in (state.active, state.previous, state.shadow) if item
        ]
        if len({item.release_id for item in pointers}) != len(pointers):
            raise ValueError("active, previous, and shadow releases must be distinct")
        return state


class ReleaseStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    def read(self) -> ReleaseState:
        try:
            value = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read release state: {error}") from error
        return ReleaseState.parse(value)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _write(self, state: ReleaseState) -> ReleaseState:
        payload = json.dumps(asdict(state), indent=2, sort_keys=True) + "\n"
        with NamedTemporaryFile(
            "w", dir=self.path.parent, prefix=f".{self.path.name}.", delete=False
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary_path.unlink(missing_ok=True)
        return state

    def initialize(self, active: ReleasePointer) -> ReleaseState:
        with self._locked():
            if self.path.exists():
                raise ValueError("release state already exists")
            return self._write(ReleaseState(1, active))

    def stage(self, shadow: ReleasePointer, expected_generation: int) -> ReleaseState:
        with self._locked():
            state = self.read()
            self._expect(state, expected_generation)
            if shadow.release_id in {
                state.active.release_id,
                state.previous.release_id if state.previous else None,
            }:
                raise ValueError("shadow release must differ from active and previous")
            return self._write(
                ReleaseState(state.generation + 1, state.active, state.previous, shadow)
            )

    def promote(self, expected_generation: int) -> ReleaseState:
        with self._locked():
            state = self.read()
            self._expect(state, expected_generation)
            if state.shadow is None:
                raise ValueError("no shadow release is staged")
            return self._write(
                ReleaseState(state.generation + 1, state.shadow, state.active)
            )

    def rollback(self, expected_generation: int) -> ReleaseState:
        with self._locked():
            state = self.read()
            self._expect(state, expected_generation)
            if state.previous is None:
                raise ValueError("no previous release is available")
            return self._write(
                ReleaseState(state.generation + 1, state.previous, state.active)
            )

    @staticmethod
    def _expect(state: ReleaseState, expected: int) -> None:
        if state.generation != expected:
            raise ValueError(
                f"stale release state: expected generation {expected}, "
                f"found {state.generation}"
            )


def _resolve_pointer(
    prepared_manifest: Path,
    tracking_root: Path,
    model_version: int,
    dense_timeout_ms: int,
) -> ReleasePointer:
    with TemporaryDirectory(prefix="scifact-release-check-") as directory:
        release = load_release(
            prepared_manifest,
            tracking_root,
            model_version,
            dense_timeout_ms,
            Path(directory),
        )
    return ReleasePointer(model_version, dense_timeout_ms, release.metadata.release_id)


def _manifest(value: Path | None, parser: argparse.ArgumentParser) -> Path:
    if value is not None:
        return value
    manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
    if len(manifests) != 1:
        parser.error("run make prepare or select one snapshot with --manifest")
    return manifests[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tracking-root", type=Path, default=TRACKING_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.set_defaults(write=False)
    for name in ("init", "stage"):
        command = commands.add_parser(name)
        command.add_argument("--model-version", type=int, required=True)
        command.add_argument("--dense-timeout-ms", type=int, default=300)
        if name == "stage":
            command.add_argument("--expected-generation", type=int, required=True)
    for name in ("promote", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--expected-generation", type=int, required=True)
    args = parser.parse_args()
    store = ReleaseStore(args.state)
    if args.command == "show":
        state = store.read()
    elif args.command == "init":
        pointer = _resolve_pointer(
            _manifest(args.manifest, parser),
            args.tracking_root,
            args.model_version,
            args.dense_timeout_ms,
        )
        state = store.initialize(pointer)
    elif args.command == "stage":
        pointer = _resolve_pointer(
            _manifest(args.manifest, parser),
            args.tracking_root,
            args.model_version,
            args.dense_timeout_ms,
        )
        state = store.stage(pointer, args.expected_generation)
    elif args.command == "promote":
        state = store.promote(args.expected_generation)
    else:
        state = store.rollback(args.expected_generation)
    print(json.dumps(asdict(state), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
