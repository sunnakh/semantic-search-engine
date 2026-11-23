import unittest
from importlib.util import find_spec
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory

from src.documents import Document
from src.reranking.qwen_mlx.checkpoint import (
    LORA_PARAMETERS,
    checkpoint_hashes,
)
from src.reranking.qwen_mlx.dataset import training_examples
from src.reranking.qwen_mlx.scorer import RelevanceScorer

DENSE_AVAILABLE = (
    find_spec("torch") is not None and find_spec("sentence_transformers") is not None
)
if DENSE_AVAILABLE:
    import torch

    from src.training.contrastive import contrastive_loss


@unittest.skipUnless(DENSE_AVAILABLE, "dense extra required")
class ContrastiveLossTests(unittest.TestCase):
    def test_all_known_positives_contribute_to_the_positive_mass(self) -> None:
        logits = torch.tensor(
            [[0.0, log(2), log(3)], [log(4), 0.0, log(2)]],
            dtype=torch.float64,
            requires_grad=True,
        )
        positives = torch.tensor([[True, True, False], [False, True, False]])
        loss = contrastive_loss(logits, positives)
        self.assertAlmostEqual(loss.item(), (log(6 / 3) + log(7 / 1)) / 2)
        loss.backward()
        self.assertTrue((logits.grad[positives] < 0).all().item())
        self.assertTrue((logits.grad[~positives] > 0).all().item())


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1 if text == "no" else 2] if text in {"no", "yes"} else [10]


class QwenMlxRefactorTests(unittest.TestCase):
    def test_checkpoint_hashes_verify_qwen_config(self) -> None:
        with TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "weights.safetensors").write_bytes(b"weights")
            (model / "config.json").write_text(
                '{"model_type":"qwen3_5","quantization":{"bits":4},'
                '"text_config":{"num_hidden_layers":32,"hidden_size":4096}}',
                encoding="utf-8",
            )
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            hashes = checkpoint_hashes(model)
            self.assertEqual(
                set(hashes), {"config.json", "tokenizer.json", "weights.safetensors"}
            )

    def test_training_examples_alternate_positive_and_negative(self) -> None:
        positive = Document("d1", "Positive", "text")
        negative = Document("d2", "Negative", "text")
        groups = [("claim", positive, negative)]
        examples = training_examples(groups)
        first = next(examples)
        second = next(examples)
        self.assertEqual((first[0], first[1].document_id, first[2]), ("claim", "d1", 1))
        self.assertEqual(
            (second[0], second[1].document_id, second[2]), ("claim", "d2", 0)
        )

    def test_scorer_resolves_single_yes_and_no_tokens(self) -> None:
        scorer = RelevanceScorer(FakeTokenizer())
        self.assertEqual(scorer.labels, [1, 2])

    def test_lora_configuration_is_preserved(self) -> None:
        self.assertEqual(
            LORA_PARAMETERS,
            {
                "rank": 8,
                "scale": 16.0,
                "dropout": 0.0,
                "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            },
        )
