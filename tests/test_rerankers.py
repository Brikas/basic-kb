"""Rerankers: registry, HTTP protocols over a fake session, local alias handling."""
from __future__ import annotations

import pytest
import requests

from basic_kb.models import SearchResult
from basic_kb.rerankers import (
    RERANKER_TYPES, DeepInfraCompatibleReranker, FastEmbedReranker, JinaCompatibleReranker, build_reranker,
)


def hits(n: int) -> list[SearchResult]:
    return [SearchResult(doc=f"doc {i}", metadata={}, score=1.0 - i / 10) for i in range(n)]


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, headers, json))
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


# --- registry -----------------------------------------------------------------------------------

def test_registry_keys():
    assert set(RERANKER_TYPES) == {"local", "jina-compatible", "deepinfra-compatible"}


def test_build_reranker_unknown_type():
    with pytest.raises(ValueError, match="Unknown reranker"):
        build_reranker("nope")


def test_build_reranker_rejects_stray_option(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "k")
    with pytest.raises(ValueError, match="rejected config option"):
        build_reranker("jina-compatible", banana=1)


def test_build_reranker_drops_none_options(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "k")
    r = build_reranker("jina-compatible", model="rerank-3", top_k_param=None)
    assert isinstance(r, JinaCompatibleReranker)
    assert r.model == "rerank-3" and r.top_k_param == "top_n"


def test_cloud_reranker_needs_api_key(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JINA_API_KEY"):
        JinaCompatibleReranker()
    assert JinaCompatibleReranker(api_key="explicit").api_key == "explicit"


def test_protocol_property_is_registry_key():
    assert JinaCompatibleReranker(api_key="k").protocol == "jina-compatible"
    assert DeepInfraCompatibleReranker(api_key="k").protocol == "deepinfra-compatible"


# --- Jina-compatible ---------------------------------------------------------------------------

def test_jina_payload_and_ordering():
    r = JinaCompatibleReranker(api_key="k")
    r._session = FakeSession([FakeResp({"results": [
        {"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]})])
    out = r.rerank("q", hits(3), top_n=2)
    url, headers, payload = r._session.calls[0]
    assert url == "https://api.jina.ai/v1/rerank"
    assert headers["Authorization"] == "Bearer k"
    assert payload == {"model": "jina-reranker-v3", "query": "q",
                       "documents": ["doc 0", "doc 1", "doc 2"], "top_n": 2}
    assert [h.doc for h in out] == ["doc 2", "doc 0"]
    assert [h.rerank_score for h in out] == [0.9, 0.5]


def test_jina_top_k_param_is_configurable():
    r = JinaCompatibleReranker(api_key="k", base_url="https://voyage/v1/rerank/", top_k_param="top_k")
    r._session = FakeSession([FakeResp({"data": [{"index": 0, "relevance_score": 1.0}]})])
    r.rerank("q", hits(1), top_n=1)
    url, _, payload = r._session.calls[0]
    assert url == "https://voyage/v1/rerank"
    assert "top_k" in payload and "top_n" not in payload


def test_empty_candidates_make_no_request():
    r = JinaCompatibleReranker(api_key="k")
    r._session = FakeSession([])
    assert r.rerank("q", [], 5) == []
    assert r._session.calls == []


def test_http_error_is_not_retried():
    r = JinaCompatibleReranker(api_key="k")
    r._session = FakeSession([FakeResp({"detail": "bad key"}, status=401)])
    with pytest.raises(RuntimeError, match="401"):
        r.rerank("q", hits(2), 2)
    assert len(r._session.calls) == 1


def test_transport_failures_are_retried_then_raised():
    r = JinaCompatibleReranker(api_key="k")
    r._session = FakeSession([requests.ConnectionError("x")] * (r.RETRIES + 1))
    with pytest.raises(RuntimeError, match="unreachable"):
        r.rerank("q", hits(2), 2)
    assert len(r._session.calls) == r.RETRIES + 1


# --- DeepInfra-compatible ----------------------------------------------------------------------

def test_deepinfra_url_and_positional_scores():
    r = DeepInfraCompatibleReranker(api_key="k")
    r._session = FakeSession([FakeResp({"scores": [0.1, 0.9, 0.5]})])
    out = r.rerank("q", hits(3), top_n=2)
    url, _, payload = r._session.calls[0]
    assert url == "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-0.6B"
    assert payload == {"queries": ["q"], "documents": ["doc 0", "doc 1", "doc 2"]}
    assert [h.doc for h in out] == ["doc 1", "doc 2"]


def test_deepinfra_score_count_mismatch():
    r = DeepInfraCompatibleReranker(api_key="k")
    r._session = FakeSession([FakeResp({"scores": [0.1]})])
    with pytest.raises(RuntimeError, match="1 scores"):
        r.rerank("q", hits(2), 2)


def test_deepinfra_missing_scores():
    r = DeepInfraCompatibleReranker(api_key="k")
    r._session = FakeSession([FakeResp({"error": "x"})])
    with pytest.raises(RuntimeError, match="no 'scores'"):
        r.rerank("q", hits(2), 2)


# --- local ---------------------------------------------------------------------------------------

def test_local_reranker_alias_resolution_without_loading():
    assert FastEmbedReranker().model == "jinaai/jina-reranker-v2-base-multilingual"
    assert FastEmbedReranker("ms-marco-MiniLM").model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert FastEmbedReranker("org/x").model == "org/x"
    assert FastEmbedReranker()._encoder is None
