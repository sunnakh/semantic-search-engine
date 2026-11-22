"""Train, bundle, and evaluate a local SciFact retrieval candidate."""

import argparse
import errno
import json
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ..data.integrity import file_hashes, manifest_digest, sha256_file
from ..data.prepared import DEFAULT_PROCESSED_ROOT, load_split, load_validation
from ..evaluation.benchmark import benchmark, evaluate_rankings, write_report
from ..evaluation.metrics import CUTOFF
from ..retrieval.dense import (
    BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    DenseIndex,
    MiniLMEncoder,
    load_minilm,
)
from .contrastive import TrainingConfig, fit_candidate

OUTPUT_ROOT = Path("artifacts/models/candidates")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def train_candidate(manifest_path: Path, output_root: Path) -> Path:
    config = TrainingConfig()
    training = load_split(manifest_path, "train")
    validation = load_validation(manifest_path)
    if training.manifest_sha256 != validation.manifest_sha256:
        raise ValueError(
            "training and validation must come from the same prepared snapshot"
        )

    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    provenance = source_provenance()
    print("Fine-tuning MiniLM on SciFact training pairs", file=sys.stderr, flush=True)
    model = load_minilm()
    training_stats = fit_candidate(model, training, config)
    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".candidate-", dir=output_root) as temporary:
        staging = Path(temporary) / "candidate"
        encoder_path = staging / "encoder"
        model.save_pretrained(
            str(encoder_path), create_model_card=False, safe_serialization=True
        )
        del model

        print("Building the index from the saved encoder", file=sys.stderr, flush=True)
        encoder = MiniLMEncoder(load_minilm(encoder_path))
        index_path = staging / "index"
        DenseIndex(validation.documents, encoder).save(index_path)
        index = DenseIndex.load(index_path, encoder)
        lineage: dict[str, object] = {
            "manifest_version": 1,
            "dataset_version": validation.dataset_version,
            "input_manifest_sha256": validation.manifest_sha256,
            "source": provenance,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "runtime_versions": {
                package: version(package)
                for package in (
                    "numpy",
                    "sentence-transformers",
                    "torch",
                    "transformers",
                )
            },
            "training_config": asdict(config),
            "training_stats": training_stats,
            "excluded_exact_test_overlap_query_ids": list(
                training.exact_test_overlap_ids
            ),
            "inference": {
                "device": "cpu",
                "batch_size": BATCH_SIZE,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "dimensions": index.dimensions,
                "similarity": "exact_cosine",
            },
            "model_and_index_sha256": file_hashes(staging),
        }
        candidate_id = manifest_digest(lineage)

        print(
            "Evaluating the saved candidate and its two references",
            file=sys.stderr,
            flush=True,
        )
        rankings = index.rank_queries(
            [validation.queries[qid] for qid in sorted(validation.queries)],
            top_k=CUTOFF,
        )
        candidate = evaluate_rankings(validation, rankings, "fine_tuned_minilm")
        candidate["model"] = {
            "candidate_id": candidate_id,
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
        }
        reports = {
            "candidate": candidate,
            "bm25": benchmark(manifest_path, "bm25"),
            "frozen_minilm": benchmark(manifest_path, "dense"),
        }
        for name, report in reports.items():
            if report["input_manifest_sha256"] != validation.manifest_sha256:
                raise ValueError(f"{name} report uses a different prepared snapshot")
            write_report(report, staging / f"{name}-validation.json")

        manifest = {
            **lineage,
            "candidate_id": candidate_id,
            "status": "evaluated",
            "validation_metrics": {
                name: report["metrics"] for name, report in reports.items()
            },
            "artifact_sha256": file_hashes(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        destination = output_root / candidate_id
        try:
            staging.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if file_hashes(destination) != file_hashes(staging):
                raise ValueError(
                    f"candidate bundle differs on rerun: {destination}"
                ) from error

    print(json.dumps(manifest["validation_metrics"], sort_keys=True))
    return destination / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="prepared SciFact manifest")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    try:
        manifest = args.manifest
        if manifest is None:
            manifests = sorted(DEFAULT_PROCESSED_ROOT.glob("*/manifest.json"))
            if len(manifests) != 1:
                raise ValueError(
                    "run make prepare or select a prepared snapshot with --manifest"
                )
            (manifest,) = manifests
        print(train_candidate(manifest, args.output_root))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Candidate training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
