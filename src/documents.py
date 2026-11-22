"""Document representation shared by data loaders and retrievers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
