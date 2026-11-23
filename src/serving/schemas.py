"""Request and response models shared by search clients and the API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..documents import Document

SEARCH_PATH = "/api/v1/search"
MAX_QUERY_LENGTH = 512
MAX_TOP_K = 50
Method = Literal["keyword", "bm25", "dense"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH
        ),
    ]
    method: Method = "dense"
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = 5


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    abstract: str

    @classmethod
    def from_document(cls, document: Document) -> "DocumentResponse":
        return cls(
            document_id=document.document_id,
            title=document.title,
            abstract=document.text,
        )


class SearchHit(DocumentResponse):
    score: float


class SearchResponse(BaseModel):
    method: Method
    model_version: str
    release_id: str
    request_id: str
    fallback: bool
    results: list[SearchHit]
    latency_ms: float
