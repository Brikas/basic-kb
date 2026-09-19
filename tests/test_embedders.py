"""Embedders: alias resolution, provider selection, and the HTTP backend over a fake transport."""
from __future__ import annotations

from dataclasses import replace

import pytest
import requests

import basic_kb.embedders as emb
from basic_kb.embedders import (
    EMBEDDER_PROVIDERS, QWEN3_QUERY_PREFIX, EmbedderBase, FastEmbedEmbedder, OpenAICompatibleEmbedder,
    build_embedder,
)

from .fakes import FakeEmbedder
from basic_kb.errors import EmbeddingError


# --- aliases / local backend (never loads a model) --------------------------------------------

def test_alias_resolution():
    assert FastEmbedEmbedder.resolve("bge-small-en-v1.5") == "BAAI/bge-small-en-v1.5"
    assert FastEmbedEmbedder.resolve("org/custom") == "org/custom"
    assert EmbedderBase.resolve("anything") == "anything"


def test_fastembed_embedder_is_lazy():
    e = FastEmbedEmbedder("all-MiniLM-L6-v2", threads=2, batch_size=0)
    assert e.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert e._model is None
    assert e._batch_size == 1          # floor


# --- build_embedder ----------------------------------------------------------------------------

def test_build_embedder_local_default(config):
    e = build_embedder(config, threads=3)
    assert isinstance(e, FastEmbedEmbedder)
    assert e._threads == 3
    assert e._batch_size == config.embed_batch_size


def test_build_embedder_api_requires_base_url(config):
    cfg = replace(config, embedding={"provider": "openai-compatible"})
    with pytest.raises(ValueError, match="base_url"):
        build_embedder(cfg)


def test_build_embedder_api_model_id_includes_dimensions(config):
    cfg = replace(config, embedding_model="Qwen/Qwen3-Embedding-8B",
                  embedding={"provider": "openai-compatible", "base_url": "https://x/v1", "dimensions": 1024})
    e = build_embedder(cfg)
    assert isinstance(e, OpenAICompatibleEmbedder)
    assert e.model_id == "Qwen/Qwen3-Embedding-8B@1024"
    assert e._query_prefix == QWEN3_QUERY_PREFIX     # default for Qwen3 when unset
    assert e._api_key_env == "EMBEDDING_API_KEY"


def test_build_embedder_unknown_provider_lists_options(config):
    with pytest.raises(ValueError, match="openai-compatible"):
        build_embedder(replace(config, embedding={"provider": "magic"}))


def test_registered_provider_is_selectable_from_config(config, monkeypatch):
    monkeypatch.setitem(EMBEDDER_PROVIDERS, "fake", lambda cfg, threads=None: FakeEmbedder(model_id=cfg.embedding_model))
    e = build_embedder(replace(config, embedding={"provider": "FAKE"}))
    assert isinstance(e, FakeEmbedder) and e.model_id == "fake-bow-64"


# --- OpenAI-compatible transport -------------------------------------------------------------

class Resp:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def api(**kw) -> OpenAICompatibleEmbedder:
    defaults = dict(model="m", base_url="https://api.test/v1/", api_key_env="TEST_EMB_KEY",
                    batch_size=64, timeout=1, max_retries=3, concurrency=1)
    defaults.update(kw)
    return OpenAICompatibleEmbedder(**defaults)


def ok_payload(inputs, dim=3, reverse=False):
    items = [{"index": i, "embedding": [float(i)] * dim} for i in range(len(inputs))]
    return {"data": list(reversed(items)) if reverse else items}


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("TEST_EMB_KEY", "sk-test")
    monkeypatch.setattr(emb.time, "sleep", lambda s: None)   # retries must not slow the suite


@pytest.fixture
def transport(monkeypatch):
    """Replace requests.post with a scripted fake. `script` is a list of Resp or exceptions;
    each call pops the next. `calls` records (url, json payload, headers)."""
    state = {"script": [], "calls": []}

    def fake_post(url, json=None, headers=None, timeout=None):
        state["calls"].append((url, json, headers))
        nxt = state["script"].pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(requests, "post", fake_post)
    return state


def test_missing_api_key_raises(monkeypatch, transport):
    monkeypatch.delenv("TEST_EMB_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="TEST_EMB_KEY"):
        api().embed(["a"])
    assert transport["calls"] == []


def test_output_is_reordered_by_index(key, transport):
    inputs = ["a", "b", "c"]
    transport["script"] = [Resp(200, ok_payload(inputs, reverse=True))]
    out = api().embed(inputs)
    assert out == [[0.0] * 3, [1.0] * 3, [2.0] * 3]
    url, payload, headers = transport["calls"][0]
    assert url == "https://api.test/v1/embeddings"
    assert payload["input"] == inputs and payload["model"] == "m"
    assert headers["Authorization"] == "Bearer sk-test"


def test_retries_on_429_then_succeeds(key, transport):
    transport["script"] = [Resp(429, text="slow down"), Resp(503), Resp(200, ok_payload(["a"]))]
    assert api().embed(["a"]) == [[0.0] * 3]
    assert len(transport["calls"]) == 3


def test_client_error_fails_immediately(key, transport):
    transport["script"] = [Resp(400, text="bad request")]
    with pytest.raises(EmbeddingError, match="400"):
        api().embed(["a"])
    assert len(transport["calls"]) == 1


def test_transport_errors_exhaust_retries(key, transport):
    transport["script"] = [requests.ConnectionError("down")] * 3
    with pytest.raises(EmbeddingError, match="after 3 attempts"):
        api().embed(["a"])


def test_wrong_vector_count_raises(key, transport):
    transport["script"] = [Resp(200, ok_payload(["a"]))]
    with pytest.raises(EmbeddingError, match="1 vectors for 2 inputs"):
        api().embed(["a", "b"])


def test_dimension_mismatch_raises(key, transport):
    transport["script"] = [Resp(200, ok_payload(["a"], dim=3))]
    with pytest.raises(EmbeddingError, match="dimensions"):
        api(dimensions=8).embed(["a"])
    assert transport["calls"][0][1]["dimensions"] == 8


def test_batches_preserve_order_sequential(key, transport):
    inputs = [f"t{i}" for i in range(5)]
    transport["script"] = [Resp(200, ok_payload(b)) for b in (inputs[0:2], inputs[2:4], inputs[4:5])]
    out = api(batch_size=2).embed(inputs)
    assert len(out) == 5
    assert [c[1]["input"] for c in transport["calls"]] == [inputs[0:2], inputs[2:4], inputs[4:5]]


def test_batches_preserve_order_concurrent(key, monkeypatch):
    # A thread-safe fake: the vector encodes the text so order can be checked.
    def fake_post(url, json=None, headers=None, timeout=None):
        data = [{"index": i, "embedding": [float(t[1:])]} for i, t in enumerate(json["input"])]
        return Resp(200, {"data": data})
    monkeypatch.setattr(requests, "post", fake_post)
    inputs = [f"t{i}" for i in range(20)]
    out = api(batch_size=3, concurrency=8).embed(inputs)
    assert out == [[float(i)] for i in range(20)]


def test_prefixes_apply_per_side(key, transport):
    transport["script"] = [Resp(200, ok_payload(["x"])), Resp(200, ok_payload(["x"]))]
    e = api(query_prefix="Q: ", passage_prefix="P: ")
    e.embed(["x"])
    e.query_embed(["x"])
    assert transport["calls"][0][1]["input"] == ["P: x"]
    assert transport["calls"][1][1]["input"] == ["Q: x"]


def test_empty_input_makes_no_request(key, transport):
    assert api().embed([]) == []
    assert transport["calls"] == []


def test_missing_fastembed_names_the_extra(monkeypatch):
    """The on-device backend is optional; its absence must say how to fix it."""
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **kw):
        if name.startswith("fastembed"):
            raise ImportError("no fastembed")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(EmbeddingError, match=r"basic-kb\[local\]"):
        FastEmbedEmbedder("bge-small-en-v1.5").embed(["x"])
