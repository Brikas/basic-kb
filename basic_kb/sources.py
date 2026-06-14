"""Data sources: discover files and parse them into ParsedDocuments.

Built-in source types (selected via `type:` in an instance config):
  transcript — speaker-tagged .md transcripts (tl;dv style)
  markdown   — .md files with optional YAML frontmatter (web copy, notes, docs)

Each source maps to its own ChromaDB collection named by its `id`.
Add a new type by subclassing DataSourceBase and registering it in SOURCE_TYPES.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .chunkers import ChunkerBase, build_chunker
from .models import ParsedDocument


def parse_transcript(path: Path) -> tuple[str, str, str]:
    """
    Parse a tl;dv transcript .md file.
    Returns (title, date, body) where body is speaker paragraphs joined by double newlines.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = lines[0].lstrip("#").strip() if lines else path.stem
    date_match = re.search(r"\*(\d{4}-\d{2}-\d{2})\*", text)
    date = date_match.group(1) if date_match else "unknown"
    paragraphs = [
        line.strip() for line in lines
        if line.strip().startswith("**") and ":**" in line
    ]
    return title, date, "\n\n".join(paragraphs)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file. Returns (metadata_dict, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm = text[3:end].strip()
    body = text[end + 4:].strip()
    try:
        import yaml as _yaml
        meta = _yaml.safe_load(fm) or {}
    except Exception:
        meta = {}
    return meta, body


# ---------------------------------------------------------------------------
# DataSource abstractions
# ---------------------------------------------------------------------------

class DataSourceBase(ABC):
    """A named source of documents that can be indexed and searched.

    Subclasses receive id/label/description/chunker from the instance config,
    so the same class can back many differently-named sources.
    """

    type_name: str = ""  # value matched against `type:` in config; set by subclass

    def __init__(
        self,
        source_id: str,
        directory: Path,
        label: str = "",
        description: str = "",
        chunker: str = "recursive",
    ) -> None:
        self._id = source_id
        self._dir = Path(directory)
        self._label = label or source_id
        self._description = description
        self._chunker = chunker

    @property
    def source_id(self) -> str:
        """Short identifier used as the ChromaDB collection name and --source value."""
        return self._id

    @property
    def label(self) -> str:
        """Human-readable name for logging."""
        return self._label

    @property
    def description(self) -> str:
        return self._description

    @property
    def directory(self) -> Path:
        return self._dir

    @abstractmethod
    def get_files(self) -> list[Path]:
        """Return all indexable files for this source."""

    @abstractmethod
    def parse_file(self, path: Path) -> Optional[ParsedDocument]:
        """Parse one file into a ParsedDocument. Return None to skip the file."""

    def make_chunker(self, chunk_size: int, overlap: int, min_chunk: int) -> ChunkerBase:
        """Build this source's chunker from its configured name."""
        return build_chunker(self._chunker, chunk_size, overlap, min_chunk)


class TranscriptSource(DataSourceBase):
    """Speaker-tagged transcript .md files (tl;dv style). Default chunker: recursive."""

    type_name = "transcript"

    def __init__(self, source_id: str, directory: Path, label: str = "",
                 description: str = "", chunker: str = "recursive") -> None:
        super().__init__(source_id, directory, label, description, chunker)

    def get_files(self) -> list[Path]:
        return sorted(self._dir.glob("*.md")) if self._dir.exists() else []

    def parse_file(self, path: Path) -> Optional[ParsedDocument]:
        title, date, body = parse_transcript(path)
        if not body:
            return None
        return ParsedDocument(
            id=path.stem,
            title=title,
            date=date,
            body=body,
            base_metadata={
                "title": title, "date": date, "file": path.name, "source": self._id,
            },
        )


class MarkdownSource(DataSourceBase):
    """
    .md files with optional YAML frontmatter. Default chunker: breadcrumb (heading-aware).

    Frontmatter `title`, `url`, `content_type`, `last_fetched` are lifted into metadata
    when present. Optional content_type filtering (e.g. "marketing", "help_center").
    """

    type_name = "markdown"

    def __init__(self, source_id: str, directory: Path, label: str = "",
                 description: str = "", chunker: str = "breadcrumb",
                 content_type_filter: Optional[str] = None) -> None:
        super().__init__(source_id, directory, label, description, chunker)
        self._ct_filter = content_type_filter

    @property
    def label(self) -> str:
        suffix = f" [{self._ct_filter}]" if self._ct_filter else ""
        return f"{self._label}{suffix}"

    def get_files(self) -> list[Path]:
        if not self._dir.exists():
            return []
        # rglob picks up files in subfolders (e.g. dictionary-words/)
        files = sorted(self._dir.rglob("*.md"))
        if not self._ct_filter:
            return files
        # Filter by content_type in frontmatter
        result = []
        for f in files:
            meta, _ = _parse_frontmatter(f.read_text(encoding="utf-8"))
            if meta.get("content_type") == self._ct_filter:
                result.append(f)
        return result

    def parse_file(self, path: Path) -> Optional[ParsedDocument]:
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        if not body.strip():
            return None
        title = str(meta.get("title", path.stem))
        return ParsedDocument(
            id=path.stem,
            title=title,
            date=str(meta.get("last_fetched", "unknown")),
            body=body,
            base_metadata={
                "title": title,
                "url": str(meta.get("url", "")),
                "content_type": str(meta.get("content_type", "unknown")),
                "file": path.name,
                "source": self._id,
            },
        )


# ---------------------------------------------------------------------------
# Source registry + factory
# ---------------------------------------------------------------------------

SOURCE_TYPES: dict[str, type[DataSourceBase]] = {
    cls.type_name: cls for cls in (TranscriptSource, MarkdownSource)
}


def build_source(cfg: dict, base_dir: Path, content_type_filter: Optional[str] = None) -> DataSourceBase:
    """Build a DataSource from one entry in a config's `sources:` list.

    `path` is resolved relative to base_dir (the config file's directory).
    Example cfg entry:
        {id: notes, type: markdown, path: data/notes, chunker: recursive,
         label: "My notes", description: "..."}
    """
    type_name = cfg.get("type", "markdown")
    cls = SOURCE_TYPES.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown source type {type_name!r} for source {cfg.get('id')!r}. "
            f"Options: {', '.join(sorted(SOURCE_TYPES))}"
        )
    raw_path = Path(cfg["path"])
    directory = raw_path if raw_path.is_absolute() else base_dir / raw_path
    kwargs: dict = dict(
        source_id=cfg["id"],
        directory=directory,
        label=cfg.get("label", ""),
        description=cfg.get("description", ""),
    )
    if "chunker" in cfg:
        kwargs["chunker"] = cfg["chunker"]
    if cls is MarkdownSource and content_type_filter:
        kwargs["content_type_filter"] = content_type_filter
    return cls(**kwargs)
