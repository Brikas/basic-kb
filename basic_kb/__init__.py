"""basic-kb — a small, config-driven semantic search engine over markdown/text.

Public API — everything returns data; nothing in this package prints.

    from basic_kb import KnowledgeBase, load_config, build_source, FastEmbedEmbedder

    kb.search(...)  -> list[SearchResult]      raises IndexNotFound / QueryFailed
    kb.index(...)   -> IndexResult             pass on_progress= for a live feed
    kb.status(...)  -> list[SourceStatus]
    kb.scan(...)    -> ScanResult

Progress and prompts are the caller's business: `index` takes `on_progress` and
`on_confirm` callbacks. The library logs to the "basic_kb" logger and writes to
no stream of its own.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings

# Windows consoles default to cp1252; our output uses →, ─, ⚠. Force UTF-8 so
# printing results never raises UnicodeEncodeError. (No-op where already UTF-8.)
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")

# Silence HF Hub / tokenizer download noise before any heavy import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
# Scoped to the noisy dependencies, NOT global. A blanket filter here silences
# every UserWarning in whatever process imports basic_kb — including deprecation
# and misuse warnings from unrelated libraries in a host application.
for _noisy in ("fastembed", "huggingface_hub", "onnxruntime", "chromadb", "transformers"):
    warnings.filterwarnings("ignore", category=UserWarning, module=rf"{_noisy}.*")

# basic-kb's own event logger. Quiet by default (NullHandler drops records when no
# log_file is configured); the CLI attaches a FileHandler when `log_file` is set.
logging.getLogger("basic_kb").addHandler(logging.NullHandler())

from .config import Config, load_config, load_env_file
from .core import KnowledgeBase, ScanResult
from .errors import (
    BasicKBError, IndexNotFound, ManifestCorrupt, MassChangeRefused, QueryFailed,
)
from .embedders import EmbedderBase, FastEmbedEmbedder
from .models import Chunk, FileError, IndexResult, ParsedDocument, SearchResult, SourceStatus
from .rerankers import JinaReranker, RerankerBase
from .sources import DataSourceBase, MarkdownSource, TranscriptSource, build_source

__version__ = "0.1.0"

__all__ = [
    "Config", "load_config", "load_env_file",
    "KnowledgeBase", "ScanResult",
    "BasicKBError", "IndexNotFound", "ManifestCorrupt", "MassChangeRefused", "QueryFailed",
    "EmbedderBase", "FastEmbedEmbedder",
    "RerankerBase", "JinaReranker",
    "Chunk", "ParsedDocument", "SearchResult",
    "FileError", "IndexResult", "SourceStatus",
    "DataSourceBase", "MarkdownSource", "TranscriptSource", "build_source",
    "__version__",
]
