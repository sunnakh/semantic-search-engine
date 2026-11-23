"""Train the local 4-bit Qwen3.5-9B checkpoint to rerank SciFact papers."""

import argparse
import json
import random
import sys
from collections.abc import Iterator
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from typing import Any, cast

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters

from ..data.integrity import sha256_file
from ..data.prepared import (
    DEFAULT_PROCESSED_ROOT,
    PreparedSplit,
    load_split,
    load_validation,
)
from ..documents import Document
from ..evaluation.benchmark import BenchmarkReport, evaluate_rankings, write_report
from ..evaluation.gate import verified_metrics
from ..evaluation.metrics import CUTOFF
from ..retrieval.lexical import BM25Index

DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
OUTPUT_ROOT = Path("artifacts/models/qwen-9b-qlora")
SEED = 2026
LAYERS = 1
LORA_PARAMETERS = {
    "rank": 8,
    "scale": 16.0,
    "dropout": 0.0,
    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
}
LEARNING_RATE = 5e-5
DEFAULT_STEPS = 100
PROMPT_VERSION = 1
INSTRUCTION = (
    "Does this scientific paper provide relevant evidence for the claim? "
    "Reply with only yes or no."
)


def checkpoint_hashes(model_path: Path) -> dict[str, str]:
    config = json.loads((model_path / "config.json").read_text())
    if (
        config.get("model_type") != "qwen3_5"
        or config.get("quantization", {}).get("bits") != 4
    ):
        raise ValueError("expected the Qwen3.5 4-bit MLX checkpoint")
    text = config["text_config"]
    if text["num_hidden_layers"] != 32 or text["hidden_size"] != 4096:
        raise ValueError("expected the 9B Qwen3.5 text backbone")
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise ValueError("the local MLX checkpoint has no safetensors weights")
    files = [model_path / "config.json", model_path / "tokenizer.json", *weights]
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        file = model_path / name
        if file.is_file():
            files.append(file)
    return {file.name: sha256_file(file) for file in files}


class RelevanceScorer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.labels = []
        for label in ("no", "yes"):
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"expected a single Qwen token for {label}")
            self.labels.append(encoded[0])

    def tokens(self, query: str, paper: Document) -> mx.array:
        claim = self.tokenizer.encode(query, add_special_tokens=False)[:64]
        passage = self.tokenizer.encode(
            f"{paper.title}\n{paper.text}", add_special_tokens=False
        )[:130]
        content = (
            f"{INSTRUCTION}\n\nClaim: {self.tokenizer.decode(claim)}\n"
            f"Paper: {self.tokenizer.decode(passage)}"
        )
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return mx.array([ids])

    def label_logits(self, model: nn.Module, tokens: mx.array) -> mx.array:
        logits = model(tokens)[0, -1]
        return mx.stack([logits[self.labels[0]], logits[self.labels[1]]]).astype(
            mx.float32
        )


def training_groups(data: PreparedSplit) -> list[tuple[str, Document, Document]]:
    documents = {document.document_id: document for document in data.documents}
    index = BM25Index(data.documents)
    excluded = set(data.exact_test_overlap_ids)
    groups = []
    for query_id in sorted(data.queries.keys() - excluded):
        query = data.queries[query_id]
        relevant = data.relevant_ids[query_id]
        negatives = [
            hit.document_id
            for hit in index.search(query, top_k=50)
            if hit.document_id not in relevant
        ]
        if negatives:
            for offset, document_id in enumerate(sorted(relevant)):
                groups.append(
                    (
                        query,
                        documents[document_id],
                        documents[negatives[offset % len(negatives)]],
                    )
                )
    if not groups:
        raise ValueError("no SciFact training claims with BM25 negatives")
    return groups


def training_examples(
    groups: list[tuple[str, Document, Document]],
) -> Iterator[tuple[str, Document, int]]:
    rng = random.Random(SEED)
    while True:
        rng.shuffle(groups)
        for query, positive, negative in groups:
            yield query, positive, 1
            yield query, negative, 0


def train(
    prepared: Path,
    model_path: Path,
    output_root: Path,
    steps: int | None = None,
    *,
    epochs: int | None = None,
) -> Path:
    hashes = checkpoint_hashes(model_path)
    data = load_split(prepared, "train")
    groups = training_groups(data)
    samples = training_examples(groups)
    steps_per_epoch = 2 * len(groups)
    total_steps = (
        epochs * steps_per_epoch if epochs is not None else steps or DEFAULT_STEPS
    )

    mx.random.seed(SEED)
    print(f"Loading local Qwen3.5-9B from {model_path}", file=sys.stderr, flush=True)
    loaded = load(str(model_path))
    model, tokenizer = loaded[0], loaded[1]
    model.freeze()
    linear_to_lora_layers(model, LAYERS, LORA_PARAMETERS)
    # Dropout is zero. Evaluation mode keeps the frozen attention kernels on
    # their inference path while MLX differentiates the final-layer adapters.
    model.eval()
    trainable = cast(
        list[tuple[str, mx.array]], tree_flatten(model.trainable_parameters())
    )
    if not trainable or any(".lora_" not in name for name, _ in trainable):
        raise RuntimeError("expected only LoRA adapter tensors to be trainable")
    parameter_count = sum(value.size for _, value in trainable)
    print(
        f"Training {parameter_count:,} QLoRA parameters for {total_steps:,} steps "
        f"({steps_per_epoch:,} examples per epoch)",
        file=sys.stderr,
        flush=True,
    )
    scorer = RelevanceScorer(tokenizer)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)

    def loss_fn(model: nn.Module, tokens: mx.array, label: int) -> mx.array:
        logits = scorer.label_logits(model, tokens)
        return mx.logsumexp(logits) - logits[label]

    value_and_grad = nn.value_and_grad(model, loss_fn)
    loss_sum = 0.0
    epoch_loss = 0.0
    epoch_history = []
    started = perf_counter()
    for step in range(1, total_steps + 1):
        query, paper, label = next(samples)
        loss, grads = value_and_grad(model, scorer.tokens(query, paper), label)
        loss_value = loss.item()
        if not isfinite(loss_value):
            raise RuntimeError("Qwen training produced a non-finite loss")
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
        if not isfinite(grad_norm.item()):
            raise RuntimeError("Qwen training produced a non-finite gradient")
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        loss_sum += loss_value
        epoch_loss += loss_value
        epoch_index, example_index = divmod(step - 1, steps_per_epoch)
        epoch, epoch_step = epoch_index + 1, example_index + 1
        epoch_label = (
            f"Epoch {epoch}/{epochs}" if epochs is not None else f"Epoch {epoch}"
        )
        finished_epoch = epoch_step == steps_per_epoch
        if epoch_step == 1 or step % 10 == 0 or finished_epoch or step == total_steps:
            print(
                f"{epoch_label} · step {epoch_step:,}/{steps_per_epoch:,} · loss {loss_value:.4f}",
                file=sys.stderr,
                flush=True,
            )
        if finished_epoch or step == total_steps:
            mean_loss = epoch_loss / epoch_step
            epoch_history.append(
                {
                    "epoch": epoch,
                    "steps": epoch_step,
                    "mean_loss": mean_loss,
                    "completed": finished_epoch,
                }
            )
            status = "complete" if finished_epoch else "partial"
            print(
                f"{epoch_label} {status} · {epoch_step:,}/{steps_per_epoch:,} examples "
                f"· mean training loss {mean_loss:.4f}",
                file=sys.stderr,
                flush=True,
            )
            epoch_loss = 0.0

    output_root.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="run-", dir=output_root))
    adapter = run / "adapter"
    adapter.mkdir()
    weights = adapter / "adapters.safetensors"
    config = adapter / "adapter_config.json"
    mx.save_safetensors(str(weights), dict(tree_flatten(model.trainable_parameters())))
    config.write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": LAYERS,
                "lora_parameters": LORA_PARAMETERS,
            },
            indent=2,
        )
        + "\n"
    )
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_version": data.dataset_version,
                "input_manifest_sha256": data.manifest_sha256,
                "base_model_path": str(model_path),
                "base_model_sha256": hashes,
                "adapter_sha256": {
                    file.name: sha256_file(file) for file in (weights, config)
                },
                "prompt_version": PROMPT_VERSION,
                "seed": SEED,
                "learning_rate": LEARNING_RATE,
                "optimizer_steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "epoch_history": epoch_history,
                "training_groups": len(groups),
                "trainable_parameters": parameter_count,
                "mean_training_loss": loss_sum / total_steps,
                "training_seconds": perf_counter() - started,
                "mlx_lm_version": version("mlx-lm"),
                "mlx_version": version("mlx"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


def evaluate(
    prepared: Path,
    run: Path,
    model_path: Path | None,
    top_n: int,
    query_limit: int | None,
    candidate_report: Path | None = None,
) -> Path:
    data = load_validation(prepared)
    candidate_rows: dict[str, list[str]] = {}
    if candidate_report is not None:
        if top_n != CUTOFF:
            raise ValueError(
                f"the Qwen embedding report uses a fixed top-{CUTOFF} cutoff"
            )
        _, source = verified_metrics(
            candidate_report, data, "frozen_qwen3_embedding_0_6b"
        )
        candidate_rows = {
            row["query_id"]: row["ranked_document_ids"] for row in source["per_query"]
        }
        if any(len(set(row)) != len(row) for row in candidate_rows.values()):
            raise ValueError("the candidate report contains duplicate document IDs")

    manifest = json.loads(run.read_text())
    adapter = run.parent / "adapter"
    model_path = (
        (model_path or Path(manifest["base_model_path"])).expanduser().resolve()
    )
    if (
        manifest["input_manifest_sha256"] != data.manifest_sha256
        or manifest["prompt_version"] != PROMPT_VERSION
        or manifest["base_model_sha256"] != checkpoint_hashes(model_path)
        or any(
            sha256_file(adapter / name) != manifest["adapter_sha256"][name]
            for name in ("adapters.safetensors", "adapter_config.json")
        )
    ):
        raise ValueError(
            "the prepared data, checkpoint, or adapter does not match the run"
        )
    if query_limit is not None:
        ids = sorted(data.queries)[:query_limit]
        data.queries = {key: data.queries[key] for key in ids}
        data.relevant_ids = {key: data.relevant_ids[key] for key in ids}

    documents = {paper.document_id: paper for paper in data.documents}
    query_ids = sorted(data.queries)
    queries = [data.queries[key] for key in query_ids]
    if candidate_report is None:
        index = BM25Index(data.documents)
        candidates = [
            [hit.document_id for hit in index.search(query, top_k=top_n)]
            for query in queries
        ]
        baseline = "bm25"
    else:
        candidates = [candidate_rows[key] for key in query_ids]
        baseline = "qwen_embedding"
    loaded = load(str(model_path))
    model, tokenizer = loaded[0], loaded[1]
    model.eval()
    scorer = RelevanceScorer(tokenizer)

    def rerank(label: str) -> list[list[str]]:
        rankings = []
        for number, (query, document_ids) in enumerate(
            zip(queries, candidates, strict=True), 1
        ):
            scores = []
            for document_id in document_ids:
                logits = scorer.label_logits(
                    model, scorer.tokens(query, documents[document_id])
                )
                confidence = cast(float, (logits[1] - logits[0]).item())
                if not isfinite(confidence):
                    raise RuntimeError("Qwen reranker returned a non-finite score")
                scores.append(confidence)
            order = sorted(
                range(len(document_ids)), key=lambda index: (-scores[index], index)
            )
            rankings.append([document_ids[index] for index in order[:CUTOFF]])
            if number == 1 or number % 10 == 0 or number == len(queries):
                print(
                    f"{label} · {number}/{len(queries)} claims",
                    file=sys.stderr,
                    flush=True,
                )
        return rankings

    reports: dict[str, BenchmarkReport] = {
        baseline: evaluate_rankings(
            data, [row[:CUTOFF] for row in candidates], baseline
        )
    }
    if candidate_report is None:
        frozen = rerank("Frozen Qwen")
        reports["frozen_qwen"] = evaluate_rankings(
            data, frozen, "frozen_qwen35_9b_reranker"
        )

    model = load_adapters(model, str(adapter))
    model.eval()
    tuned = rerank("QLoRA Qwen")
    reports["qlora_qwen"] = evaluate_rankings(data, tuned, "qlora_qwen35_9b_reranker")
    suffix = (
        "validation" if query_limit is None else f"validation-sample-{len(queries)}"
    )
    if candidate_report is not None:
        suffix = f"qwen-embedding-{suffix}"
    for name, report in reports.items():
        write_report(report, run.parent / f"{name}-{suffix}.json")
    comparison = run.parent / f"comparison-{suffix}.json"
    result: dict[str, object] = {
        "validation_queries": len(queries),
        "requested_subset": query_limit,
        "candidate_depth": top_n,
        **{method: report["metrics"] for method, report in reports.items()},
    }
    if candidate_report is not None:
        result["candidate_report_sha256"] = sha256_file(candidate_report)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    comparison.write_text(serialized)
    print(serialized, end="")
    return comparison


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
    if not mx.metal.is_available():
        parser.error("Apple Silicon Metal is required; use a native macOS terminal")
    if args.command == "evaluate" and args.top_n < CUTOFF:
        parser.error(f"--top-n must be at least {CUTOFF}")
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
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        print(f"Qwen3.5-9B {args.command} failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
