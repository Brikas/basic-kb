"""Deterministic stand-ins for the model-backed parts of the engine.

The suite never downloads or loads a real model. These fakes keep the *shape* of
the real backends (same ABCs, same return types) while making results predictable:
texts that share words come out closer in vector space, so a search test can
assert ranking rather than a mocked number.
"""
from __future__ import annotations

import hashlib
import math
import re

from basic_kb.embedders import EmbedderBase
from basic_kb.models import SearchResult
from basic_kb.rerankers import RerankerBase

_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder(EmbedderBase):
    """Bag-of-words hashed into a fixed-size vector, L2-normalised.

    Each word is hashed to one dimension (sha1 of the word, mod `dim`); a vector is
    the normalised count of words per dimension. Two texts sharing words therefore
    have a higher cosine than two that share none, which is all a ranking test
    needs. `calls` records every batch so a test can assert what was embedded.
    """

    def __init__(self, dim: int = 64, model_id: str = "fake-bow-64") -> None:
        self.dim = dim
        self._model_id = model_id
        self.calls: list[list[str]] = []          # every embed() batch, in order
        self.query_calls: list[list[str]] = []    # every query_embed() batch

    @property
    def model_id(self) -> str:
        return self._model_id

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            slot = int(hashlib.sha1(w.encode()).hexdigest(), 16) % self.dim
            v[slot] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            v[0] = 1.0     # empty text: a fixed unit vector, never NaN
            return v
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        self.query_calls.append(list(texts))
        return [self._vec(t) for t in texts]


class ExplodingEmbedder(FakeEmbedder):
    """Raises on any embed call. Used to prove a code path never touched the model
    (for example the CLI attaching to a served instance)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embedder was called; this path must not embed")

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embedder was called; this path must not embed")


class FakeReranker(RerankerBase):
    """Scores each hit by how many query words its text contains."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []    # (query, candidate count)

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        self.calls.append((query, len(results)))
        qwords = set(_WORD.findall(query.lower()))
        for r in results:
            r.rerank_score = float(len(qwords & set(_WORD.findall(r.doc.lower()))))
        return sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)[:top_n]


class FailingReranker(RerankerBase):
    """Always raises, to exercise the strict/fallback branches."""

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        raise RuntimeError("reranker down")
