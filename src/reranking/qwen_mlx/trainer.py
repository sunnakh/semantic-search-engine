"""QLoRA training loop for the local Qwen reranker."""

import json
import importlib
import subprocess
import sys
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from typing import cast

from ...data.integrity import manifest_digest, sha256_file
from ...data.prepared import load_split
from .checkpoint import (
    DEFAULT_STEPS,
    LAYERS,
    LEARNING_RATE,
    LORA_PARAMETERS,
    PROMPT_VERSION,
    SEED,
    checkpoint_hashes,
)
from .dataset import training_examples, training_groups
from .scorer import RelevanceScorer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def source_provenance() -> dict[str, str | None]:
    code_commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            code_commit = commit.stdout.strip()
    except FileNotFoundError:
        pass
    source = PROJECT_ROOT / "src"
    source_hash = manifest_digest(
        {
            path.relative_to(source).as_posix(): sha256_file(path)
            for path in sorted(source.rglob("*.py"))
        }
    )
    return {
        "commit": code_commit,
        "python_source_sha256": source_hash,
        "lockfile_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
    }


def train(
    prepared: Path,
    model_path: Path,
    output_root: Path,
    steps: int | None = None,
    *,
    epochs: int | None = None,
) -> Path:
    mx = importlib.import_module("mlx.core")
    optim = importlib.import_module("mlx.optimizers")
    mlx_utils = importlib.import_module("mlx.utils")
    mlx_lm = importlib.import_module("mlx_lm")
    tuner_utils = importlib.import_module("mlx_lm.tuner.utils")
    nn = importlib.import_module("mlx.nn")
    tree_flatten = mlx_utils.tree_flatten
    load = mlx_lm.load
    linear_to_lora_layers = tuner_utils.linear_to_lora_layers

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
        + "\n",
        encoding="utf-8",
    )
    manifest = run / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_version": data.dataset_version,
                "input_manifest_sha256": data.manifest_sha256,
                "base_model_path": str(model_path),
                "base_model_sha256": hashes,
                "source": source_provenance(),
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
        + "\n",
        encoding="utf-8",
    )
    return manifest
