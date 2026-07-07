"""basic-kb — a small, config-driven semantic search engine over markdown/text.

Public API:
    from basic_kb import KnowledgeBase, load_config, build_source, FastEmbedEmbedder
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
warnings.filterwarnings("ignore", category=UserWarning)

# basic-kb's own event logger. Quiet by default (NullHandler drops records when no
# log_file is configured); the CLI attaches a FileHandler when `log_file` is set.
logging.getLogger("basic_kb").addHandler(logging.NullHandler())

from .config import Config, load_config, load_env_file
from .core import KnowledgeBase
from .embedders import EmbedderBase, FastEmbedEmbedder
from .models import Chunk, ParsedDocument, SearchResult
from .rerankers import JinaReranker, RerankerBase
from .sources import DataSourceBase, MarkdownSource, TranscriptSource, build_source

__version__ = "0.1.0"

__all__ = [
    "Config", "load_config", "load_env_file",
    "KnowledgeBase",
    "EmbedderBase", "FastEmbedEmbedder",
    "RerankerBase", "JinaReranker",
    "Chunk", "ParsedDocument", "SearchResult",
    "DataSourceBase", "MarkdownSource", "TranscriptSource", "build_source",
    "__version__",
]
