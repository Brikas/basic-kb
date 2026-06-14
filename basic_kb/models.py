"""Core data models passed between sources, chunkers, and the knowledge base."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    doc: str
    metadata: dict
    score: float
    rerank_score: Optional[float] = None

    @property
    def sort_key(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass
class ParsedDocument:
    """A parsed source document ready for chunking."""
    id: str              # unique stable ID used to build chunk IDs
    title: str
    date: str
    body: str
    base_metadata: dict  # attached to every chunk produced from this doc


@dataclass
class Chunk:
    """A single indexed unit with its text and metadata."""
    text: str
    metadata: dict
    id_suffix: str  # combined with doc.id → "{doc.id}::{id_suffix}"
