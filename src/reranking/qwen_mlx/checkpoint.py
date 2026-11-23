"""Qwen MLX checkpoint validation and constants."""

import json
from pathlib import Path

from ...data.integrity import sha256_file

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
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
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
