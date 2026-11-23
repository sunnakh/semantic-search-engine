"""Train and evaluate a LoRA adapter for Qwen3-Embedding-0.6B on SciFact."""

import argparse
import errno
import json
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from numpy.typing import NDArray
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer

from ..data.integrity import file_hashes, manifest_digest
from ..data.prepared import (
    DEFAULT_PROCESSED_ROOT,
    PreparedSplit,
    load_split,
    load_validation,
)
from ..evaluation.benchmark import BenchmarkReport, evaluate_rankings, write_report
from ..evaluation.gate import verified_metrics
from ..evaluation.metrics import CUTOFF
from ..retrieval.dense import MODEL_CACHE, DenseIndex
from .candidate import source_provenance
from .contrastive import TrainingConfig, fit_candidate

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QUERY_PROMPT = (
    "Instruct: Given a scientific claim, retrieve scientific paper abstracts "
    "that provide relevant evidence for the claim.\nQuery: "
)
MAX_SEQ_LENGTH = 256
EMBEDDING_DIMENSIONS = 1024
EVAL_BATCH_SIZE = 16
EVAL_CHUNK_SIZE = 256
FROZEN_METHOD = "frozen_qwen3_embedding_0_6b"
ADAPTED_METHOD = "lora_qwen3_embedding_0_6b"
OUTPUT_ROOT = Path("artifacts/models/qwen-lora")
FROZEN_REPORT = Path("artifacts/evaluation/qwen3-embedding-0.6b/qwen-validation.json")


def load_qwen(device: str, *, training: bool = False) -> SentenceTransformer:
    dtype = torch.float16 if device == "mps" and not training else torch.float32
    model = SentenceTransformer(
        MODEL_ID,
        revision=REVISION,
        device=device,
        cache_folder=str(MODEL_CACHE),
        trust_remote_code=False,
        model_kwargs={"dtype": dtype, "use_safetensors": True},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def report_progress(step: int, loss: float) -> None:
    if step == 1 or step % 10 == 0:
        print(f"Step {step:3d} · loss {loss:.4f}", file=sys.stderr, flush=True)


def train(
    prepared: Path,
    output_root: Path,
    device: str,
    config: TrainingConfig,
    max_steps: int | None,
) -> Path:
    data = load_split(prepared, "train")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    model = load_qwen(device, training=True)
    adapter = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model.add_adapter(adapter)
    trainable = [
        (name, p.numel()) for name, p in model.named_parameters() if p.requires_grad
    ]
    if not trainable or any(".lora_" not in name for name, _ in trainable):
        raise RuntimeError("expected only LoRA parameters to be trainable")
    count = sum(size for _, size in trainable)
    print(
        f"Training {count:,} LoRA parameters on {device} with FP32 weights",
        file=sys.stderr,
        flush=True,
    )
    stats = fit_candidate(
        model,
        data,
        config,
        query_prompt=QUERY_PROMPT,
        max_steps=max_steps,
        on_step=report_progress,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".qwen-lora-", dir=output_root) as temporary:
        staging = Path(temporary) / "run"
        adapter_path = staging / "adapter"
        model.save_pretrained(
            str(adapter_path), create_model_card=False, safe_serialization=True
        )
        if (
            not (adapter_path / "adapter_model.safetensors").is_file()
            or (adapter_path / "model.safetensors").exists()
        ):
            raise RuntimeError("expected an adapter-only checkpoint")
        source = source_provenance()
        source["checkout_head"] = source.pop("commit")
        metadata = {
            "manifest_version": 1,
            "dataset_version": data.dataset_version,
            "input_manifest_sha256": data.manifest_sha256,
            "base_model": {"id": MODEL_ID, "revision": REVISION},
            "source": source,
            "training_config": asdict(config),
            "training_stats": stats,
            "max_steps": max_steps,
            "device": device,
            "training_dtype": "float32",
            "trainable_parameters": count,
            "query_prompt": QUERY_PROMPT,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "excluded_exact_test_overlap_query_ids": list(data.exact_test_overlap_ids),
            "runtime_versions": {
                package: version(package)
                for package in (
                    "peft",
                    "numpy",
                    "torch",
                    "transformers",
                    "sentence-transformers",
                )
            },
            "adapter_sha256": file_hashes(adapter_path),
        }
        run_id = manifest_digest(metadata)
        (staging / "manifest.json").write_text(
            json.dumps({**metadata, "run_id": run_id}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination = output_root / run_id
        try:
            staging.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if file_hashes(destination) != file_hashes(staging):
                raise ValueError(
                    f"existing adapter does not match the run ID: {destination}"
                ) from error
    print(json.dumps(stats, sort_keys=True))
    return destination / "manifest.json"


@dataclass(frozen=True)
class QwenEncoder:
    model: SentenceTransformer
    prompt: str | None = None

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        chunks = []
        for start in range(0, len(texts), EVAL_CHUNK_SIZE):
            batch = texts[start : start + EVAL_CHUNK_SIZE]
            vectors = self.model.encode(
                batch,
                prompt=self.prompt,
                batch_size=EVAL_BATCH_SIZE,
                show_progress_bar=False,
            )
            chunks.append(np.asarray(vectors, dtype=np.float32))
            if len(texts) > EVAL_CHUNK_SIZE:
                print(
                    f"Embedded {start + len(batch):,}/{len(texts):,} abstracts",
                    file=sys.stderr,
                    flush=True,
                )
        return np.concatenate(chunks)


def score(
    data: PreparedSplit,
    device: str,
    adapter_path: Path | None = None,
) -> BenchmarkReport:
    model = load_qwen(device)
    if adapter_path is not None:
        model.load_adapter(str(adapter_path))
    model.eval()
    index = DenseIndex(
        data.documents,
        QwenEncoder(model),
        query_encoder=QwenEncoder(model, QUERY_PROMPT),
    )
    if index.dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError("unexpected Qwen embedding dimension")
    rankings = index.rank_queries(
        [data.queries[qid] for qid in sorted(data.queries)], top_k=CUTOFF
    )
    report = evaluate_rankings(
        data, rankings, ADAPTED_METHOD if adapter_path else FROZEN_METHOD
    )
    report["model"] = inference_settings(device)
    return report


def inference_settings(device: str) -> dict[str, int | str]:
    return {
        "id": MODEL_ID,
        "revision": REVISION,
        "device": device,
        "dtype": "float16" if device == "mps" else "float32",
        "query_prompt": QUERY_PROMPT,
        "max_sequence_length": MAX_SEQ_LENGTH,
        "batch_size": EVAL_BATCH_SIZE,
        "dimensions": EMBEDDING_DIMENSIONS,
        "similarity": "exact_cosine",
        "sentence_transformers_version": version("sentence-transformers"),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
    }


def evaluate(prepared: Path, run_path: Path, frozen_path: Path, device: str) -> Path:
    data = load_validation(prepared)
    manifest = json.loads(run_path.read_text())
    run_id = manifest.pop("run_id")
    adapter_path = run_path.parent / "adapter"
    if (
        run_id != manifest_digest(manifest)
        or manifest["dataset_version"] != data.dataset_version
        or manifest["input_manifest_sha256"] != data.manifest_sha256
        or manifest["base_model"] != {"id": MODEL_ID, "revision": REVISION}
        or manifest["query_prompt"] != QUERY_PROMPT
        or manifest["max_sequence_length"] != MAX_SEQ_LENGTH
        or manifest["adapter_sha256"] != file_hashes(adapter_path)
    ):
        raise ValueError(
            "adapter, training manifest, or prepared snapshot did not match"
        )

    if frozen_path.is_file():
        frozen_metrics, frozen_report = verified_metrics(
            frozen_path, data, FROZEN_METHOD
        )
        if inference_settings(device) != frozen_report.get("model"):
            raise ValueError(
                "frozen report settings differ; pass a fresh --frozen-report path"
            )
    else:
        print("Scoring frozen Qwen on validation", file=sys.stderr, flush=True)
        frozen = score(data, device)
        write_report(frozen, frozen_path)
        frozen_metrics = frozen["metrics"]
    print("Scoring the saved LoRA adapter on validation", file=sys.stderr, flush=True)
    adapted = score(data, device, adapter_path)
    write_report(adapted, run_path.parent / "validation.json")
    comparison = {
        "run_id": run_id,
        "validation_queries": adapted["query_count"],
        "frozen_qwen": frozen_metrics,
        "lora_qwen": adapted["metrics"],
        "ndcg_at_10_delta": adapted["metrics"]["ndcg@10"] - frozen_metrics["ndcg@10"],
    }
    path = run_path.parent / "comparison.json"
    serialized = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return path


def positive_int(text: str) -> int:
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
        command.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    trainer.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    trainer.add_argument("--epochs", type=positive_int, default=1)
    trainer.add_argument("--batch-size", type=positive_int, default=8)
    trainer.add_argument(
        "--max-steps", type=positive_int, help="limit optimizer steps for a smoke run"
    )
    evaluator.add_argument(
        "--run", type=Path, required=True, help="LoRA training manifest"
    )
    evaluator.add_argument("--frozen-report", type=Path, default=FROZEN_REPORT)
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error(
            "Apple GPU support is unavailable. Use a native terminal or --device cpu."
        )
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
            if args.batch_size < 2:
                parser.error("--batch-size must be at least 2 for contrastive training")
            config = TrainingConfig(
                epochs=args.epochs, batch_size=args.batch_size, learning_rate=5e-5
            )
            result = train(prepared, args.output_root, device, config, args.max_steps)
        else:
            result = evaluate(prepared, args.run, args.frozen_report, device)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        print(f"Qwen LoRA {args.command} failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
