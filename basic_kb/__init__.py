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
import warnings

# Silence HF Hub / tokenizer download noise before any heavy import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
# Scoped to the noisy dependencies, NOT global. A blanket filter here silences
# every UserWarning in whatever process imports basic_kb — including deprecation
# and misuse warnings from unrelated libraries in a host application.
for _noisy in ("fastembed", "huggingface_hub", "onnxruntime", "transformers"):
    warnings.filterwarnings("ignore", category=UserWarning, module=rf"{_noisy}.*")

# basic-kb's own event logger. Quiet by default (NullHandler drops records when no
# log_file is configured); the CLI attaches a FileHandler when `log_file` is set.
logging.getLogger("basic_kb").addHandler(logging.NullHandler())

from .config import Config, load_config, load_env_file
from .core import KnowledgeBase
from .store import SqliteVecStore, VacuumPolicy
from .errors import (
    BasicKBError, EmbeddingError, IndexNotFound, MassChangeRefused, QueryFailed, StoreError, UnknownSource,
)
from .embedders import EMBEDDER_PROVIDERS, EmbedderBase, FastEmbedEmbedder, OpenAICompatibleEmbedder, build_embedder
from .models import (
    Chunk, FileError, IndexResult, InstanceInfo, ParsedDocument, ReindexResult, ScanResult, SearchResult,
    SourceInfo, SourceStatus,
)
from .rerankers import RERANKER_TYPES, DeepInfraCompatibleReranker, JinaCompatibleReranker, RerankerBase, build_reranker
from .serialize import from_dict, to_jsonable
from .sources import DataSourceBase, MarkdownSource, TranscriptSource, build_source, resolve_sources

from .version import __version__  # noqa: E402


def open(config_path=None, **kwargs):  # noqa: A001 - `basic_kb.open(...)` reads as intended
    """A local KnowledgeBase for an instance: `basic_kb.open("path/to/basic-kb.yaml")`.

    With no path the config is discovered like the CLI does ($BASIC_KB_CONFIG, then a
    basic-kb.yaml up from the working directory). The config's env_file is loaded so
    provider API keys are available. Extra keyword arguments go to
    `KnowledgeBase.from_config` (model=, threads=, reranker=, strict_reranker=, on_warning=).
    """
    from pathlib import Path

    from .config import find_config

    path = Path(config_path).expanduser() if config_path else find_config()
    if path is None:
        raise FileNotFoundError("no basic-kb.yaml found: pass a path, set BASIC_KB_CONFIG, "
                                "or run from inside an instance folder")
    config = load_config(path)
    if config.env_file and config.env_file.exists():
        load_env_file(config.env_file)
    return KnowledgeBase.from_config(config, **kwargs)


def connect(url, api_key=None, **kwargs):
    """A served instance over HTTP: `basic_kb.connect("http://host:8765", api_key)`.

    Same methods and return types as a local KnowledgeBase. The key falls back to
    `BASIC_KB_API_KEY` when omitted.
    """
    from .client import RemoteKnowledgeBase, resolve_api_key

    return RemoteKnowledgeBase(url, api_key=resolve_api_key(api_key), **kwargs)

__all__ = [
    "Config", "load_config", "load_env_file",
    "KnowledgeBase", "SqliteVecStore", "VacuumPolicy",
    "BasicKBError", "EmbeddingError", "IndexNotFound", "MassChangeRefused", "QueryFailed", "StoreError",
    "UnknownSource",
    "EMBEDDER_PROVIDERS", "EmbedderBase", "FastEmbedEmbedder", "OpenAICompatibleEmbedder", "build_embedder",
    "RERANKER_TYPES", "RerankerBase", "JinaCompatibleReranker", "DeepInfraCompatibleReranker", "build_reranker",
    "Chunk", "ParsedDocument", "SearchResult",
    "FileError", "IndexResult", "InstanceInfo", "ReindexResult", "ScanResult", "SourceInfo", "SourceStatus",
    "from_dict", "to_jsonable",
    "DataSourceBase", "MarkdownSource", "TranscriptSource", "build_source", "resolve_sources",
    "open", "connect",
    "__version__",
]
