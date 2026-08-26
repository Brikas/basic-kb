"""Core data models passed between sources, chunkers, and the knowledge base."""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class FileError:
    """One file that could not be processed during an index run.

    Collected rather than raised so a single undecodable file does not abandon a
    run that has already embedded hundreds of others.
    """
    rel_path: str
    error: str


@dataclass
class IndexResult:
    """What an index run actually did. Returned by `KnowledgeBase.index`.

    `embedded` is the work done this run; `unchanged` is what the manifest let it
    skip. `aborted` is True when the mass-change guard stopped the run — check it
    rather than inferring success from the absence of an exception.
    """
    source_id: str
    label: str
    files_on_disk: int
    added: int = 0            # files indexed for the first time
    updated: int = 0          # files re-embedded after changing
    unchanged: int = 0        # skipped — hash matched the manifest
    empty: int = 0            # parsed to no chunks (tracked, not indexed)
    pruned: int = 0           # removed because they vanished from disk
    total_chunks: int = 0     # chunks in the collection after the run
    limited_to: Optional[int] = None   # the `limit` in force, if any
    aborted: bool = False
    abort_reason: Optional[str] = None
    errors: list[FileError] = field(default_factory=list)

    @property
    def embedded(self) -> int:
        return self.added + self.updated


@dataclass
class SourceStatus:
    """Index stats for one source. Returned by `KnowledgeBase.status`.

    Every field a caller might want to render, computed once. The CLI formats it;
    nothing here prints.
    """
    source_id: str
    label: str
    directory: str
    store_dir: str
    model_id: str
    directory_exists: bool
    indexed: bool                      # a collection exists for this source
    chunks: int = 0
    docs_with_chunks: int = 0
    files_on_disk: int = 0
    tracked: bool = False              # a manifest entry exists
    new: int = 0
    updated: int = 0
    deleted: int = 0
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    content_types: dict[str, int] = field(default_factory=dict)
    oversized_chunks: int = 0

    @property
    def stale(self) -> int:
        return self.new + self.updated + self.deleted
