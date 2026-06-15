"""Rerankers — optional second-stage cross-encoder reordering of search hits.

Backends are pluggable via RERANKER_TYPES; a config's `reranker:` selects one:
  local — FastEmbed cross-encoder, on-device, no API key, private (default choice)
  jina  — Jina cloud API, needs JINA_API_KEY

Add a new backend: subclass RerankerBase, register it in RERANKER_TYPES.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from .models import SearchResult


class RerankerBase(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]: ...


class JinaReranker(RerankerBase):
    """
    Reranker backed by the Jina AI reranking API.

    Supported models:
      jina-reranker-v3               — SOTA BEIR 61.94, 131K context (recommended)
      jina-reranker-v2-base-multilingual — BEIR 57.06, 1K context (auto-chunked)
      jina-reranker-v1-base-en       — BEIR 55.88, Apache 2.0, free self-host
    """

    API_URL = "https://api.jina.ai/v1/rerank"

    def __init__(self, model: str = "jina-reranker-v3", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")
        if not self.api_key:
            raise ValueError("Jina API key not found. Set JINA_API_KEY env var or pass api_key=.")

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return results
        import requests as _requests
        docs = [r.doc for r in results]
        try:
            resp = _requests.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "query": query, "documents": docs, "top_n": top_n},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except _requests.HTTPError as e:
            raise RuntimeError(f"Jina rerank API error {e.response.status_code}: {e.response.text}") from e

        reranked: list[SearchResult] = []
        for item in data.get("results", []):
            r = results[item["index"]]
            r.rerank_score = item["relevance_score"]
            reranked.append(r)
        return sorted(reranked, key=lambda r: r.rerank_score or 0, reverse=True)[:top_n]


class FastEmbedReranker(RerankerBase):
    """Local cross-encoder reranker via FastEmbed (ONNX, on-device, no API key).

    The model is downloaded once and cached, then loaded lazily on first rerank
    (so constructing the reranker is cheap). Any TextCrossEncoder model id works;
    a few short aliases are provided.
    """

    DEFAULT_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
    ALIASES = {
        "jina-v2-multilingual": "jinaai/jina-reranker-v2-base-multilingual",
        "jina-v1-turbo": "jinaai/jina-reranker-v1-turbo-en",
        "bge-base": "BAAI/bge-reranker-base",
        "ms-marco-MiniLM": "Xenova/ms-marco-MiniLM-L-6-v2",
    }

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = self.ALIASES.get(model or "", model) or self.DEFAULT_MODEL
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._encoder = TextCrossEncoder(self.model)

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return results
        self._load()
        # rerank() yields one relevance score per document, in input order.
        scores = list(self._encoder.rerank(query, [r.doc for r in results]))
        for r, s in zip(results, scores):
            r.rerank_score = float(s)
        return sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)[:top_n]


# Pluggable backends — `reranker:` in a config selects one by key.
RERANKER_TYPES: dict[str, type[RerankerBase]] = {
    "local": FastEmbedReranker,
    "jina": JinaReranker,
}


def build_reranker(rtype: str, model: Optional[str] = None) -> RerankerBase:
    """Build a reranker by type. Raises on unknown type or missing prerequisites
    (e.g. jina without an API key) — the caller decides whether to fall back."""
    cls = RERANKER_TYPES.get(rtype.lower())
    if cls is None:
        raise ValueError(
            f"Unknown reranker {rtype!r}. Options: {', '.join(sorted(RERANKER_TYPES))}, none."
        )
    return cls(model) if model else cls()
