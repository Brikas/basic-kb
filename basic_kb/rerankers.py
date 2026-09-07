"""Rerankers — optional second-stage cross-encoder reordering of search hits.

Backends are pluggable via RERANKER_TYPES; a config's `reranker:` selects one.
A key names a *wire protocol*, not a vendor, because several vendors speak each:
  local                — FastEmbed cross-encoder, on-device, no API key, private
  jina-compatible      — `/v1/rerank`: Jina, Voyage, SiliconFlow
  deepinfra-compatible — DeepInfra native `/v1/inference/<model>`

Point a protocol at a vendor with `base_url:`, `api_key_env:` and `model:` in the
config block, the same way `embedding:` selects an embedding provider.

Add a protocol: subclass RerankAPIBase for anything bearer-auth over HTTP (it
brings the Session, retries and error handling), else subclass RerankerBase.
Register it in RERANKER_TYPES — the CLI's `--reranker` choices follow the registry.
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


class RerankAPIBase(RerankerBase):
    """Shared HTTP plumbing for cloud rerankers: bearer auth, one pooled Session,
    and retries limited to transient transport failures.

    Subclasses name a *wire protocol*, not a vendor — several vendors speak each
    one. A config picks the protocol with `type:` and the vendor with `base_url:`,
    `api_key_env:` and `model:`.
    """

    DEFAULT_BASE_URL: str = ""
    DEFAULT_API_KEY_ENV: str = ""
    DEFAULT_MODEL: str = ""

    TIMEOUT_S = 15
    RETRIES = 2          # total attempts = RETRIES + 1

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key_env: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or self.DEFAULT_MODEL
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env or self.DEFAULT_API_KEY_ENV
        self.api_key = api_key or os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"{self.protocol} API key not found. "
                f"Set {self.api_key_env} env var or pass api_key=."
            )
        # One Session for the life of the reranker. Without it every search pays a
        # fresh TCP connect and TLS handshake — on the hot path, per query.
        self._session = None
        self._session_lock = threading.Lock()

    @property
    def protocol(self) -> str:
        """Registry key this class is registered under, for error messages and logs."""
        for key, cls in RERANKER_TYPES.items():
            if cls is type(self):
                return key
        return type(self).__name__

    def _get_session(self):
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                import requests as _requests
                self._session = _requests.Session()
            return self._session

    def _post(self, url: str, payload: dict) -> dict:
        """POST with bearer auth, returning the decoded JSON body.

        Retries only transient transport failures. An HTTP error is a real answer
        (bad key, bad model, rate limit) — retrying it just delays the report.
        """
        import requests as _requests
        session = self._get_session()
        last_exc: Optional[Exception] = None
        for attempt in range(self.RETRIES + 1):
            try:
                resp = session.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.TIMEOUT_S,
                )
                resp.raise_for_status()
                return resp.json()
            except _requests.HTTPError as e:
                raise RuntimeError(
                    f"{self.protocol} rerank API error "
                    f"{e.response.status_code}: {e.response.text}") from e
            except (_requests.Timeout, _requests.ConnectionError) as e:
                last_exc = e
                logger.warning("%s rerank transport failure (attempt %d/%d): %s",
                               self.protocol, attempt + 1, self.RETRIES + 1, e)
        raise RuntimeError(
            f"{self.protocol} rerank unreachable after "
            f"{self.RETRIES + 1} attempts: {last_exc}"
        ) from last_exc


class JinaCompatibleReranker(RerankAPIBase):
    """`POST <base_url>` with `{model, query, documents, <top-k>}`, answering with
    `results[]` of `{index, relevance_score}`.

    Spoken by Jina, Voyage and SiliconFlow. Vendors differ only in base_url, key
    env var, model, and the top-k parameter name — Jina spells it `top_n`, Voyage
    `top_k`, so `top_k_param:` is a per-vendor config knob.

    Jina models: jina-reranker-v3 (SOTA multilingual), jina-reranker-v2-base-multilingual.
    Voyage models: rerank-3, rerank-3-lite (both 32K context, multilingual).
    """

    DEFAULT_BASE_URL = "https://api.jina.ai/v1/rerank"
    DEFAULT_API_KEY_ENV = "JINA_API_KEY"
    DEFAULT_MODEL = "jina-reranker-v3"
    DEFAULT_TOP_K_PARAM = "top_n"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key_env: Optional[str] = None, api_key: Optional[str] = None,
                 top_k_param: Optional[str] = None) -> None:
        super().__init__(model=model, base_url=base_url,
                         api_key_env=api_key_env, api_key=api_key)
        self.top_k_param = top_k_param or self.DEFAULT_TOP_K_PARAM

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return results
        payload = {"model": self.model, "query": query,
                   "documents": [r.doc for r in results], self.top_k_param: top_n}
        data = self._post(self.base_url, payload)

        reranked: list[SearchResult] = []
        # Jina and Voyage return the scored list under "results"; some use "data".
        for item in data.get("results") or data.get("data") or []:
            r = results[item["index"]]
            r.rerank_score = item["relevance_score"]
            reranked.append(r)
        return sorted(reranked, key=lambda r: r.rerank_score or 0, reverse=True)[:top_n]


class DeepInfraCompatibleReranker(RerankAPIBase):
    """`POST <base_url>/<model>` with `{queries: [q], documents: [...]}`, answering
    with `scores[]` positionally aligned to the documents sent.

    DeepInfra's native inference shape: the model is a URL path segment, the single
    query broadcasts across every document, and there is no top-k parameter — the
    full candidate pool comes back scored and this class trims it.

    Models: Qwen/Qwen3-Reranker-0.6B (multilingual, Apache 2.0), BAAI/bge-reranker-v2-m3.
    """

    DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/inference"
    DEFAULT_API_KEY_ENV = "DEEPINFRA_API_KEY"
    DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return results
        payload = {"queries": [query], "documents": [r.doc for r in results]}
        data = self._post(f"{self.base_url}/{self.model}", payload)

        scores = data.get("scores")
        if scores is None:
            raise RuntimeError(f"{self.protocol} rerank response has no 'scores': {data}")
        if len(scores) != len(results):
            raise RuntimeError(
                f"{self.protocol} rerank returned {len(scores)} scores "
                f"for {len(results)} documents")
        for r, s in zip(results, scores):
            r.rerank_score = float(s)
        return sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)[:top_n]


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


# Pluggable backends — `reranker:` in a config selects one by key. Keys name the
# wire protocol, so one entry serves every vendor that speaks it.
RERANKER_TYPES: dict[str, type[RerankerBase]] = {
    "local": FastEmbedReranker,
    "jina-compatible": JinaCompatibleReranker,
    "deepinfra-compatible": DeepInfraCompatibleReranker,
}


def build_reranker(rtype: str, model: Optional[str] = None, **options) -> RerankerBase:
    """Build a reranker by protocol key. Extra `options` come from the config's
    `reranker:` block (base_url, api_key_env, top_k_param) and are passed to the
    backend's constructor.

    Raises on unknown type, an option the backend does not accept, or missing
    prerequisites (a cloud backend without an API key) — the caller decides
    whether to fall back."""
    cls = RERANKER_TYPES.get(rtype.lower())
    if cls is None:
        raise ValueError(
            f"Unknown reranker {rtype!r}. Options: {', '.join(sorted(RERANKER_TYPES))}, none."
        )
    kwargs = {k: v for k, v in options.items() if v is not None}
    if model:
        kwargs["model"] = model
    try:
        return cls(**kwargs)
    except TypeError as e:
        # A stray key in the config block would otherwise surface as a bare
        # "unexpected keyword argument" with no hint of where it came from.
        raise ValueError(f"reranker {rtype!r} rejected config option: {e}") from e
