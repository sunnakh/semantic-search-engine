"""MiniLM embeddings and exact cosine retrieval."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DenseHit:
    document_id: str
    score: float


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


def load_minilm(directory: Path | None = None) -> SentenceTransformer:
    if directory is not None and not directory.is_dir():
        raise ValueError(f"saved encoder directory is missing: {directory}")
    model = SentenceTransformer(
        MODEL_ID if directory is None else str(directory),
        revision=MODEL_REVISION if directory is None else None,
        local_files_only=directory is not None,
        device="cpu",
        cache_folder=str(MODEL_CACHE),
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    return model


def embedding_text(document: Document) -> str:
    return f"{document.title} {document.text}".strip()


class MiniLMEncoder:
    def __init__(self, model: SentenceTransformer | None = None) -> None:
        self._model = load_minilm() if model is None else model

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
    def __init__(
        self,
        documents: Sequence[Document],
        encoder: Encoder,
        *,
        query_encoder: Encoder | None = None,
    ) -> None:
        if not documents:
            raise ValueError("dense index requires at least one document")
        ordered = sorted(documents, key=lambda document: document.document_id)
        self._ids = [document.document_id for document in ordered]
        self._encoder = query_encoder if query_encoder is not None else encoder
        self._vectors = unit_embeddings(
            encoder,
            [embedding_text(document) for document in ordered],
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True)
        np.save(directory / "vectors.npy", self._vectors, allow_pickle=False)
        (directory / "document_ids.json").write_text(
            json.dumps(self._ids) + "\n", encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, directory: Path, encoder: Encoder) -> "DenseIndex":
        ids = json.loads((directory / "document_ids.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or sorted(set(ids)) != ids
        ):
            raise ValueError("dense index requires sorted unique document IDs")
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(ids)
            or not vectors.shape[1]
            or vectors.dtype != np.float32
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
        ):
            raise ValueError(
                "dense index requires one normalized float32 vector per document"
            )
        index = cls.__new__(cls)
        index._ids = ids
        index._vectors = vectors
        index._encoder = encoder
        return index

    @property
    def dimensions(self) -> int:
        return self._vectors.shape[1]

    def rank_queries(self, queries: list[str], top_k: int = 10) -> list[list[str]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not queries:
            return []
        _, ranked = self._rank(queries, top_k)
        return [[self._ids[index] for index in row] for row in ranked]

    def search(self, query: str, top_k: int = 10) -> list[DenseHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores, ranked = self._rank([query], top_k)
        return [
            DenseHit(self._ids[index], float(scores[0, index])) for index in ranked[0]
        ]

    def _rank(
        self, queries: list[str], top_k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
        query_vectors = unit_embeddings(self._encoder, queries)
        if query_vectors.shape[1] != self.dimensions:
            raise ValueError(
                "query and document embeddings must have the same dimension"
            )

        scores = query_vectors @ self._vectors.T
        # Columns already follow document ID order, which resolves exact score ties.
        ranked = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        return scores, ranked
