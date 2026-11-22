import unittest
from importlib.util import find_spec
from math import log

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
