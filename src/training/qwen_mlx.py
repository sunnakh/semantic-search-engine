"""Train and evaluate the local 4-bit Qwen3.5-9B reranker."""

import argparse
import importlib
import sys
from pathlib import Path

from ..data.prepared import DEFAULT_PROCESSED_ROOT
from ..reranking.qwen_mlx.evaluator import evaluate
from ..reranking.qwen_mlx.trainer import train

DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
OUTPUT_ROOT = Path("artifacts/models/qwen-9b-qlora")


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train")
    evaluator = commands.add_parser("evaluate")
    for command in (trainer, evaluator):
        command.add_argument("--manifest", type=Path, help="prepared SciFact manifest")
        command.add_argument(
            "--model", type=Path, help="local Qwen3.5-9B MLX checkpoint directory"
        )
    duration = trainer.add_mutually_exclusive_group()
    duration.add_argument(
        "--max-steps", type=positive, help="optimizer step budget (default: 100)"
    )
    duration.add_argument(
        "--epochs",
        type=positive,
        help="number of full passes through the training examples",
    )
    trainer.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    evaluator.add_argument(
        "--run", type=Path, required=True, help="training run manifest"
    )
    evaluator.add_argument("--top-n", type=positive, default=10)
    evaluator.add_argument(
        "--candidate-report",
        type=Path,
        help="reuse the frozen Qwen3-Embedding-0.6B top-10 validation rankings",
    )
    evaluator.add_argument(
        "--query-limit", type=positive, help="evaluate a small validation subset"
    )
    args = parser.parse_args()

    try:
        mx = importlib.import_module("mlx.core")
    except (ImportError, ModuleNotFoundError):
        parser.error("the qwen-mlx extra is required; use a native macOS terminal")
    if not mx.metal.is_available():
        parser.error("Apple Silicon Metal is required; use a native macOS terminal")
    if args.command == "evaluate" and args.top_n < 10:
        parser.error("--top-n must be at least 10")
    prepared = args.manifest
    if prepared is None:
        manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
        if len(manifests) != 1:
            parser.error(
                "run make prepare or select a prepared snapshot with --manifest"
            )
        (prepared,) = manifests
    try:
        if args.command == "train":
            model = (args.model or DEFAULT_MODEL).expanduser().resolve()
            result = train(
                prepared, model, args.output_root, args.max_steps, epochs=args.epochs
            )
        else:
            result = evaluate(
                prepared,
                args.run,
                args.model,
                args.top_n,
                args.query_limit,
                args.candidate_report,
            )
            if args.query_limit is None:
                import json

                comparison = json.loads(result.read_text(encoding="utf-8"))
                if not comparison.get("gate", {}).get("passed", True):
                    print(result)
                    return 2
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        print(f"Qwen3.5-9B {args.command} failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
