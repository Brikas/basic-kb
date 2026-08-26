"""Rerankers — optional second-stage cross-encoder reordering of search hits.

Backends are pluggable via RERANKER_TYPES; a config's `reranker:` selects one:
  local — FastEmbed cross-encoder, on-device, no API key, private (default choice)
  jina  — Jina cloud API, needs JINA_API_KEY

Add a new backend: subclass RerankerBase, register it in RERANKER_TYPES.
"""
from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

from .models import SearchResult

logger = logging.getLogger("basic_kb")


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
    TIMEOUT_S = 15
    RETRIES = 2          # total attempts = RETRIES + 1

    def __init__(self, model: str = "jina-reranker-v3", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")
        if not self.api_key:
            raise ValueError("Jina API key not found. Set JINA_API_KEY env var or pass api_key=.")
        # One Session for the life of the reranker. Without it every search pays a
        # fresh TCP connect and TLS handshake — on the hot path, per query.
        self._session = None
        self._session_lock = threading.Lock()

    def _get_session(self):
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                import requests as _requests
                self._session = _requests.Session()
            return self._session

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return results
        import requests as _requests
        docs = [r.doc for r in results]
        session = self._get_session()
        payload = {"model": self.model, "query": query, "documents": docs, "top_n": top_n}

        # Retry only transient transport failures. An HTTP error is a real answer
        # (bad key, bad model, rate limit) — retrying it just delays the report.
        last_exc: Optional[Exception] = None
        for attempt in range(self.RETRIES + 1):
            try:
                resp = session.post(
                    self.API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.TIMEOUT_S,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except _requests.HTTPError as e:
                raise RuntimeError(
                    f"Jina rerank API error {e.response.status_code}: {e.response.text}") from e
            except (_requests.Timeout, _requests.ConnectionError) as e:
                last_exc = e
                logger.warning("jina rerank transport failure (attempt %d/%d): %s",
                               attempt + 1, self.RETRIES + 1, e)
        else:
            raise RuntimeError(
                f"Jina rerank unreachable after {self.RETRIES + 1} attempts: {last_exc}"
            ) from last_exc

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
        self._load_lock = threading.Lock()   # same lazy-load race as the embedder

    def _load(self):
        if self._encoder is not None:
            return
        with self._load_lock:
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
