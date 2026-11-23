"""Hardware-independent prompt and MLX relevance-scoring helpers."""

import importlib
from typing import Any

from ...documents import Document
from .checkpoint import INSTRUCTION


class RelevanceScorer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.labels = []
        for label in ("no", "yes"):
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"expected a single Qwen token for {label}")
            self.labels.append(encoded[0])

    def tokens(self, query: str, paper: Document) -> Any:
        mx = importlib.import_module("mlx.core")

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

    def label_logits(self, model: Any, tokens: Any) -> Any:
        mx = importlib.import_module("mlx.core")

        logits = model(tokens)[0, -1]
        return mx.stack([logits[self.labels[0]], logits[self.labels[1]]]).astype(
            mx.float32
        )
