"""Embedding backends. Local FastEmbed/ONNX by default (no API, no network)."""
from __future__ import annotations

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

    def as_chroma_ef(self):
        import chromadb
        embedder = self

        class _ChromaEF(chromadb.EmbeddingFunction):
            def __init__(self) -> None:
                pass

            def __call__(self, input):
                return embedder.embed(input)

        ef = _ChromaEF()
        ef.__class__.__name__ = f"FastEmbed_{embedder.model_id.replace('/', '_')}"
        return ef

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
    }

    def __init__(self, alias: str = DEFAULT_MODEL, threads: Optional[int] = None) -> None:
        self._alias = alias
        self._model_id = self.SUPPORTED.get(alias, alias)
        self._threads = threads   # cap ONNX threads (CPU cores) used; None = all
        self._model: Any = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self._model_id, show_progress=False, threads=self._threads)

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return [v.tolist() for v in self._model.embed(texts)]

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return [v.tolist() for v in self._model.query_embed(texts)]
