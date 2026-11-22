"""MiniLM embeddings and exact cosine retrieval."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..documents import Document

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
MODEL_CACHE = Path("artifacts/models/huggingface")


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


class MiniLMEncoder:
    def __init__(self) -> None:
        self._model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            device="cpu",
            cache_folder=str(MODEL_CACHE),
            trust_remote_code=False,
            model_kwargs={"use_safetensors": True},
        )
        self._model.max_seq_length = MAX_SEQUENCE_LENGTH

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray(
            self._model.encode(
                texts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def unit_embeddings(encoder: Encoder, texts: list[str]) -> NDArray[np.float32]:
    embeddings = encoder.encode(texts)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(texts)
        or not embeddings.shape[1]
    ):
        raise ValueError(
            "encoder must return one vector per input with a nonzero dimension"
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("encoder returned zero or non-finite vectors")
    return embeddings / norms


class DenseIndex:
    def __init__(self, documents: Sequence[Document], encoder: Encoder) -> None:
        if not documents:
            raise ValueError("dense index requires at least one document")
        ordered = sorted(documents, key=lambda document: document.document_id)
        self._ids = [document.document_id for document in ordered]
        self._encoder = encoder
        self._vectors = unit_embeddings(
            encoder,
            [f"{document.title} {document.text}".strip() for document in ordered],
        )

    @property
    def dimensions(self) -> int:
        return self._vectors.shape[1]

    def rank_queries(self, queries: list[str], top_k: int = 10) -> list[list[str]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not queries:
            return []
        query_vectors = unit_embeddings(self._encoder, queries)
        if query_vectors.shape[1] != self.dimensions:
            raise ValueError(
                "query and document embeddings must have the same dimension"
            )

        scores = query_vectors @ self._vectors.T
        # Columns already follow document ID order, which resolves exact score ties.
        ranked = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        return [[self._ids[index] for index in row] for row in ranked]
