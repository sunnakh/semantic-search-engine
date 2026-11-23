"""Fine-tune a dual encoder on SciFact's judged query-document pairs."""

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch import Tensor
from torch.nn.functional import normalize

from ..data.prepared import PreparedSplit
from ..retrieval.dense import embedding_text


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 2026
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-5
    temperature: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    cpu_threads: int = 4


def contrastive_loss(logits: Tensor, positives: Tensor) -> Tensor:
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    return (logits.logsumexp(dim=1) - positive_logits.logsumexp(dim=1)).mean()


def fit_candidate(
    model: SentenceTransformer,
    data: PreparedSplit,
    config: TrainingConfig,
    *,
    query_prompt: str | None = None,
    max_steps: int | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> dict[str, int | float]:
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    documents = {doc.document_id: embedding_text(doc) for doc in data.documents}
    excluded = set(data.exact_test_overlap_ids)
    pairs = sorted(
        (query_id, document_id)
        for query_id, relevant in data.relevant_ids.items()
        if query_id not in excluded
        for document_id in relevant
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    device = model.device
    model.train()
    examples = steps = 0
    total_loss = 0.0

    for _ in range(config.epochs):
        rng.shuffle(pairs)
        for start in range(0, len(pairs), config.batch_size):
            batch = pairs[start : start + config.batch_size]
            query_ids = list(dict.fromkeys(query_id for query_id, _ in batch))
            document_ids = list(dict.fromkeys(document_id for _, document_id in batch))
            positives = torch.tensor(
                [
                    [doc_id in data.relevant_ids[qid] for doc_id in document_ids]
                    for qid in query_ids
                ],
                dtype=torch.bool,
            )
            if positives.all():
                continue

            optimizer.zero_grad(set_to_none=True)
            queries = model(
                batch_to_device(
                    model.preprocess(
                        [data.queries[qid] for qid in query_ids], prompt=query_prompt
                    ),
                    device,
                )
            )
            passages = model(
                batch_to_device(
                    model.preprocess([documents[doc_id] for doc_id in document_ids]),
                    device,
                )
            )
            query_vectors = normalize(queries["sentence_embedding"].float(), dim=1)
            document_vectors = normalize(passages["sentence_embedding"].float(), dim=1)
            logits = query_vectors @ document_vectors.T / config.temperature
            loss = contrastive_loss(logits, positives.to(device))
            if not torch.isfinite(loss):
                raise ValueError("non-finite contrastive training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()

            examples += len(query_ids)
            steps += 1
            loss_value = loss.item()
            total_loss += loss_value * len(query_ids)
            if on_step is not None:
                on_step(steps, loss_value)
            if steps == max_steps:
                break
        if steps == max_steps:
            break

    if not steps:
        raise ValueError(
            "training requires at least one batch with both positive and unjudged pairs"
        )
    return {
        "training_query_count": len(data.queries) - len(excluded),
        "training_pair_count": len(pairs),
        "optimizer_steps": steps,
        "mean_training_loss": total_loss / examples,
    }
