"""Chunkers: turn a ParsedDocument into a list of indexable Chunks.

  recursive  — fixed-size recursive character splitter (transcripts, plain notes)
  breadcrumb — heading-aware; prefixes each chunk with "# Page > ## Section"
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

from .models import Chunk, ParsedDocument
from .textsplit import split_text

# Chunking defaults. Override per-instance via config or per-call via CLI flags.
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 150
DEFAULT_MIN_CHUNK = 50


class ChunkerBase(ABC):
    """Turns a ParsedDocument into a list of Chunks."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...

    @property
    def oversized_warnings(self) -> list[str]:
        """Warnings about sections that exceeded max_chunk_size. Override in subclasses."""
        return []


class RecursiveChunker(ChunkerBase):
    """
    Fixed-size recursive character splitter. Good for transcripts and plain notes
    without heading structure. Breaks on paragraph → newline → sentence → word.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        min_chunk: int = DEFAULT_MIN_CHUNK,
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_chunk = min_chunk

    @property
    def name(self) -> str:
        return "RecursiveChunker"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        texts = split_text(doc.body, self._chunk_size, self._overlap, self._min_chunk)
        return [
            Chunk(
                text=t,
                metadata={**doc.base_metadata, "chunk_index": i},
                id_suffix=str(i),
            )
            for i, t in enumerate(texts)
        ]


class BreadcrumbHeadingChunker(ChunkerBase):
    """
    Heading-aware chunker for web pages.

    Each chunk is prefixed with a breadcrumb showing its position in the document:

        # Page Title > ## Section Name

        Section body text goes here...

    Rules:
    - H1 always goes to metadata, not chunk text.
    - Intro text (before first H2) → breadcrumb is just "# Page Title".
    - Each H2 + body → one chunk (if it fits within max_chunk_size).
    - Oversized sections are sub-split using split_text and flagged with
      oversized=True in metadata. A warning is printed during indexing.
    - Images lines (![...]) are stripped before chunking.
    """

    def __init__(
        self,
        max_chunk_size: int = 1200,
        min_chunk_size: int = 50,
        overlap: int = 100,
    ) -> None:
        self._max = max_chunk_size
        self._min = min_chunk_size
        self._overlap = overlap

    @property
    def name(self) -> str:
        return "BreadcrumbHeadingChunker"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        body = doc.body

        # Extract H1 → page title for breadcrumbs
        h1_match = re.search(r'^# (.+)$', body, re.MULTILINE)
        page_h1 = h1_match.group(1).strip() if h1_match else doc.title

        # Locate all H2 headings
        h2_matches = list(re.finditer(r'^## (.+)$', body, re.MULTILINE))

        chunks: list[Chunk] = []
        section_idx = 0

        if not h2_matches:
            # No H2 — whole body is one section under the H1
            cleaned = self._clean(re.sub(r'^# .+\n?', '', body, flags=re.MULTILINE))
            if len(cleaned) >= self._min:
                new = self._make_chunks(cleaned, f"# {page_h1}", page_h1, None, doc.base_metadata, section_idx)
                chunks.extend(new)
            return chunks

        # Intro section: everything before the first H2 (minus the H1 line)
        intro_raw = body[:h2_matches[0].start()]
        intro_raw = re.sub(r'^# .+\n?', '', intro_raw, flags=re.MULTILINE)
        intro = self._clean(intro_raw)
        if intro and len(intro) >= self._min:
            new = self._make_chunks(intro, f"# {page_h1}", page_h1, None, doc.base_metadata, section_idx)
            chunks.extend(new)
            section_idx += len(new)

        # H2 sections
        for i, m in enumerate(h2_matches):
            h2_heading = m.group(1).strip()
            body_start = m.end() + 1
            body_end = h2_matches[i + 1].start() if i + 1 < len(h2_matches) else len(body)
            section_body = self._clean(body[body_start:body_end])

            # Dictionary pages often have the definition as the H2 text itself with nothing after.
            # Fall back to the heading text so the page still produces a chunk.
            if not section_body or len(section_body) < self._min:
                if len(h2_heading) >= self._min:
                    section_body = h2_heading
                else:
                    continue

            breadcrumb = f"# {page_h1} > ## {h2_heading}"
            meta_extra = {**doc.base_metadata, "h1": page_h1, "h2": h2_heading}
            new = self._make_chunks(section_body, breadcrumb, page_h1, h2_heading, meta_extra, section_idx)
            chunks.extend(new)
            section_idx += len(new)

        return chunks

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Strip image lines, tracking pixels, and collapse blank lines."""
        text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)\n?', '', text)  # linked images
        text = re.sub(r'!\[.*?\]\(.*?\)\n?', '', text)              # bare images
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _make_chunks(
        self,
        section_body: str,
        breadcrumb: str,
        page_h1: str,
        h2_heading: Optional[str],
        meta: dict,
        start_idx: int,
    ) -> list[Chunk]:
        full_text = f"{breadcrumb}\n\n{section_body}"

        if len(full_text) <= self._max:
            return [Chunk(
                text=full_text,
                metadata={**meta, "breadcrumb": breadcrumb},
                id_suffix=str(start_idx),
            )]

        # Section exceeds max_chunk_size — split and carry breadcrumb into every piece
        budget = max(self._max - len(breadcrumb) - 4, 200)  # 4 = "\n\n"
        pieces = split_text(section_body, budget, self._overlap, self._min)

        return [
            Chunk(
                text=f"{breadcrumb}\n\n{piece}",
                metadata={**meta, "breadcrumb": breadcrumb},
                id_suffix=str(start_idx + j),
            )
            for j, piece in enumerate(pieces)
        ]


# ---------------------------------------------------------------------------
# Chunker registry
# ---------------------------------------------------------------------------

# Maps the `chunker:` name in a config to a uniform (size, overlap, min) factory.
# Normalizes the differing constructor signatures behind one call.
_CHUNKER_FACTORIES = {
    "recursive": lambda size, overlap, mn: RecursiveChunker(size, overlap, mn),
    "breadcrumb": lambda size, overlap, mn: BreadcrumbHeadingChunker(
        max_chunk_size=size, min_chunk_size=mn, overlap=overlap
    ),
}


def build_chunker(name: str, chunk_size: int, overlap: int, min_chunk: int) -> ChunkerBase:
    """Build a chunker by config name. Raises on unknown name (no silent fallback)."""
    factory = _CHUNKER_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown chunker {name!r}. Options: {', '.join(sorted(_CHUNKER_FACTORIES))}"
        )
    return factory(chunk_size, overlap, min_chunk)
