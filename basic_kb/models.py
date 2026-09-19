"""Core data models passed between sources, chunkers, and the knowledge base."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional


# Metadata an agent cannot act on: indexing bookkeeping, and `file`, which `rel_path`
# already contains together with the folder it sits in. Stripped unless `detailed`.
BOOKKEEPING_METADATA = frozenset({"content_hash", "position", "chunk_index", "file"})


@dataclass
class SearchResult:
    doc: str
    metadata: dict
    score: float
    rerank_score: Optional[float] = None

    @property
    def sort_key(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    def trimmed(self, detailed: bool = False, max_chars: int = 0) -> "SearchResult":
        """A copy shaped for a machine reader. `detailed` keeps every stored field;
        `max_chars` caps the chunk text (0 = full, which is the default everywhere).

        Applied at the serialisation boundary, so the CLI, the API and the client all
        return the same thing, and nothing about the stored index changes.
        """
        meta = self.metadata if detailed else {
            k: v for k, v in self.metadata.items() if k not in BOOKKEEPING_METADATA}
        doc = self.doc if not max_chars else self.doc[:max_chars]
        return SearchResult(doc=doc, metadata=meta, score=self.score, rerank_score=self.rerank_score)


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
    total_chunks: int = 0     # chunks in the store after the run
    chunks_embedded: int = 0  # chunks actually sent to the embedder this run
    chunks_reused: int = 0    # chunks whose text was unchanged — kept their vector
    limited_to: Optional[int] = None   # the `limit` in force, if any
    aborted: bool = False
    abort_reason: Optional[str] = None
    errors: list[FileError] = field(default_factory=list)

    @property
    def embedded(self) -> int:
        return self.added + self.updated


@dataclass
class ScanResult:
    """Read-only diff of a source's files on disk vs. what was last indexed."""
    source_id: str
    label: str
    tracked: bool        # False if nothing indexed yet for this source
    files_on_disk: int
    new: int
    updated: int
    unchanged: int
    deleted: int

    @property
    def stale(self) -> int:
        """Files that differ from the index (would change it on re-index)."""
        return self.new + self.updated + self.deleted


@dataclass
class ReindexResult:
    """What a targeted re-index of specific files did. Returned by `KnowledgeBase.reindex_paths`."""
    source_id: str
    embedded: int = 0         # files whose chunks were (re)embedded
    empty: int = 0            # files that parsed to no chunks (tracked, hold nothing)
    pruned: int = 0           # files removed because they vanished from disk
    unchanged: int = 0        # files whose hash already matched the manifest
    chunks_embedded: int = 0
    chunks_reused: int = 0

    @property
    def changed(self) -> bool:
        """True when the store was written (the index clock moved)."""
        return bool(self.embedded or self.empty or self.pruned)

    def summary(self) -> str:
        """`k=v` for every non-zero counter, or "no change"."""
        parts = [f"{f.name}={getattr(self, f.name)}" for f in fields(self)
                 if f.name != "source_id" and getattr(self, f.name)]
        return ", ".join(parts) or "no change"


@dataclass
class PreviewChunk:
    """One chunk as the chunker would produce it, for a dry run."""
    text: str
    breadcrumb: Optional[str] = None


@dataclass
class PreviewFile:
    """One file's dry-run chunking. `body_chars == 0` means the file parsed to nothing."""
    rel_path: str
    body_chars: int
    chunks: list[PreviewChunk] = field(default_factory=list)

    @property
    def skipped(self) -> bool:
        return self.body_chars == 0


@dataclass
class VacuumResult:
    """What `KnowledgeBase.vacuum` did, with the store's size and counters afterwards."""
    vacuumed: bool            # False when the database was busy or there is no store yet
    path: str
    size_bytes: int
    live: int                 # chunks in the store
    deleted_since_vacuum: int


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
    chars: int = 0                     # characters of embedded chunk text
    docs_with_chunks: int = 0
    files_on_disk: int = 0
    tracked: bool = False              # a manifest entry exists
    new: int = 0
    updated: int = 0
    deleted: int = 0
    indexed_at: Optional[float] = None  # epoch seconds of the last index run; None if unstamped
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    content_types: dict[str, int] = field(default_factory=dict)
    oversized_chunks: int = 0

    @property
    def stale(self) -> int:
        return self.new + self.updated + self.deleted

    @property
    def pending(self) -> int:
        """Files the next index run has to embed — created and edited alike. Splitting
        those two tells a reader nothing: both mean "not in the index yet"."""
        return self.new + self.updated

    @property
    def approx_tokens(self) -> int:
        """Rough token count of the embedded text: 4 chars per token."""
        return self.chars // 4


@dataclass
class SourceInfo:
    """What one source *is*, for a reader deciding whether to search it.

    Distinct from SourceStatus, which answers "is the index current". This answers
    "what is in here and is it worth querying" — so it carries the config's
    description, which status has no reason to show.
    """
    source_id: str
    label: str
    description: str
    type: str                  # source type from config (transcript, markdown, …)
    chunker: str
    files: int                 # files on disk
    chunks: int                # chunks in the index
    indexed: bool              # a collection exists and holds something

    @property
    def chunks_per_file(self) -> float:
        return round(self.chunks / self.files, 1) if self.files else 0.0


@dataclass
class InstanceInfo:
    """The whole knowledge base at a glance. Returned by `KnowledgeBase.info`."""
    name: str
    model_id: str
    store_dir: str
    sources: list[SourceInfo] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return sum(s.files for s in self.sources)

    @property
    def total_chunks(self) -> int:
        return sum(s.chunks for s in self.sources)
