"""Embedding backends. Local FastEmbed/ONNX by default (no API, no network).

Embedders return plain lists of floats; the store (store.py) packs them itself."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

# Default embedding model alias (see FastEmbedEmbedder.SUPPORTED for the full set).
DEFAULT_MODEL = "bge-small-en-v1.5"


class EmbedderBase(ABC):
    SUPPORTED: dict[str, str] = {}

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed query-side text. Override for asymmetric models (e.g. BGE)."""
        return self.embed(texts)

    @classmethod
    def resolve(cls, alias: str) -> str:
        return cls.SUPPORTED.get(alias, alias)


class FastEmbedEmbedder(EmbedderBase):
    """
    Embedding model backed by FastEmbed (ONNX, runs fully local).

    FastEmbed benchmark on M1 Mac (1500-char chunks):
      bge-small-en-v1.5 — 8.3 chunks/s  (~10 min for 451 files)
      bge-base-en-v1.5  — 2.5 chunks/s  (~32 min for 451 files)
      all-MiniLM-L6-v2  — 90  chunks/s  (< 1 min for 451 files, MTEB 56)
    """

    SUPPORTED = {
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        # ~50 languages (incl. Lithuanian, Danish), 384-dim, 118M params, symmetric (no prefix).
        # CAUTION: its tokenizer caps input at 128 tokens (~500 chars); fastembed honours that,
        # so anything past ~500 chars of a chunk is silently dropped. Only usable with chunks
        # sized to it (chunk_size <= ~450), not with the 1200-char default.
        "multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }

    # Texts per ONNX forward pass. Transformer attention memory scales with
    # batch x seq_len^2, and onnxruntime's arena never returns the peak to the OS, so a
    # long-lived process keeps whatever the biggest batch needed. Measured on bge-small,
    # 241 x 1200-char chunks, 2 threads: batch 256 (fastembed default) peaked at 4.2 GB,
    # batch 32 at 1.1 GB, batch 8 at 0.48 GB — all at the same throughput (CPU-bound).
    DEFAULT_BATCH_SIZE = 8

    def __init__(self, alias: str = DEFAULT_MODEL, threads: Optional[int] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._alias = alias
        self._model_id = self.SUPPORTED.get(alias, alias)
        self._threads = threads   # cap ONNX threads (CPU cores) used; None = all
        self._batch_size = max(1, int(batch_size))
        self._model: Any = None
        # Guards the lazy load. Without it, concurrent first-calls in a server each
        # pass the None check and each construct a model — an N-fold memory spike on
        # a cold process, at exactly the moment traffic arrives.
        self._load_lock = threading.Lock()

    def _load(self):
        if self._model is not None:      # fast path, no lock once loaded
            return
        with self._load_lock:
            if self._model is None:      # re-check: another thread may have won
                from fastembed import TextEmbedding
                self._model = TextEmbedding(
                    self._model_id, show_progress=False, threads=self._threads)

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return [v.tolist() for v in self._model.embed(texts, batch_size=self._batch_size)]

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return [v.tolist() for v in self._model.query_embed(texts, batch_size=self._batch_size)]
