"""Rerankers. Optional cross-encoder reranking via the Jina API (needs JINA_API_KEY)."""
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
