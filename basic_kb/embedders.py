"""Embedding backends.

Two kinds, one interface (`EmbedderBase`): local ONNX via FastEmbed (default, no
network) and any OpenAI-compatible `/v1/embeddings` HTTP endpoint (DeepInfra, Nebius,
OpenRouter, OpenAI, a self-hosted vLLM, …). Embedders return plain lists of floats;
the store (store.py) packs them itself. `build_embedder()` picks the backend from the
`embedding:` config block.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from .errors import EmbeddingError

logger = logging.getLogger("basic_kb")

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

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self._model_id, show_progress=False, threads=self._threads)

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return [v.tolist() for v in self._model.embed(texts, batch_size=self._batch_size)]

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return [v.tolist() for v in self._model.query_embed(texts, batch_size=self._batch_size)]


class OpenAICompatibleEmbedder(EmbedderBase):
    """Embeddings from any OpenAI-compatible `POST {base_url}/embeddings` endpoint.

    Works for DeepInfra, Nebius, OpenRouter, OpenAI, Voyage's OpenAI-compatible route,
    or a self-hosted vLLM/TEI. The API key is read from the environment variable named
    by `api_key_env` (never from the config file). Texts are sent in batches; a failed
    batch is retried with backoff on 429/5xx/network errors and raised otherwise —
    an embedding that silently came back empty would poison the index.

    Instruction-aware models (Qwen3-Embedding, e5, BGE) want a prefix on the query side
    and/or the passage side; `query_prefix` / `passage_prefix` add them. Passages are
    what gets stored, so changing either prefix changes the geometry — the model id
    reported to the store includes the requested dimensions so a change forces a rebuild.
    """

    def __init__(self, model: str, base_url: str, api_key_env: str,
                 dimensions: Optional[int] = None, batch_size: int = 64,
                 query_prefix: str = "", passage_prefix: str = "",
                 timeout: float = 60.0, max_retries: int = 5) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._dimensions = int(dimensions) if dimensions else None
        self._batch_size = max(1, int(batch_size))
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def model_id(self) -> str:
        # Dimensions are part of the identity: the same model at 1024 vs 4096 dims is a
        # different vector space, and the store refuses to mix spaces.
        return f"{self._model}@{self._dimensions}" if self._dimensions else self._model

    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env, "").strip()
        if not key:
            raise EmbeddingError(
                f"embedding API key missing: set {self._api_key_env} in the environment or in the "
                f"dotenv file the config points to (`env_file:`). Never put the key in the config.")
        return key

    def _post(self, inputs: list[str]) -> list[list[float]]:
        import requests

        payload: dict = {"model": self._model, "input": inputs, "encoding_format": "float"}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        url = f"{self._base_url}/embeddings"
        delay = 1.0
        for attempt in range(1, self._max_retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            except requests.RequestException as e:
                if attempt == self._max_retries:
                    raise EmbeddingError(f"embedding request failed after {attempt} attempts: {e}") from e
                logger.warning("embedding request error (%s); retry %d in %.0fs", e, attempt, delay)
                time.sleep(delay); delay = min(delay * 2, 30)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                logger.warning("embedding API %s; retry %d in %.0fs", r.status_code, attempt, delay)
                time.sleep(delay); delay = min(delay * 2, 30)
                continue
            if r.status_code != 200:
                raise EmbeddingError(f"embedding API {r.status_code} from {url}: {r.text[:300]}")
            data = r.json().get("data")
            if not isinstance(data, list) or len(data) != len(inputs):
                raise EmbeddingError(f"embedding API returned {len(data) if isinstance(data, list) else 'no'} "
                                   f"vectors for {len(inputs)} inputs")
            # The API may reorder; `index` is authoritative.
            out: list[list[float]] = [None] * len(inputs)  # type: ignore[list-item]
            for item in data:
                out[int(item["index"])] = [float(x) for x in item["embedding"]]
            if any(v is None for v in out):
                raise EmbeddingError("embedding API response missing indices")
            if self._dimensions and len(out[0]) != self._dimensions:
                raise EmbeddingError(
                    f"asked {self._model} for {self._dimensions} dimensions, got {len(out[0])} — "
                    f"this provider ignores `dimensions`; drop it from the config or pick another provider")
            return out
        raise EmbeddingError("unreachable")

    def _embed_all(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = [prefix + t for t in texts[i:i + self._batch_size]]
            out.extend(self._post(batch))
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts, self._passage_prefix)

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts, self._query_prefix)


# Qwen3-Embedding is instruction-aware: the authors recommend an English task instruction
# on the query side (+1–5% in their evals) and none on passages. This is their default
# retrieval instruction, used when a config selects a Qwen3-Embedding model over the API
# without its own `query_prefix`.
QWEN3_QUERY_PREFIX = ("Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                      "Query:")


def build_embedder(config, threads: Optional[int] = None) -> EmbedderBase:
    """Backend from the `embedding:` config block. `provider: local` (default) → FastEmbed;
    `provider: openai-compatible` → HTTP endpoint. Thread cap applies to local only."""
    emb = config.embedding
    provider = str(emb.get("provider", "local")).lower()
    model = config.embedding_model
    if provider in ("local", "fastembed"):
        return FastEmbedEmbedder(alias=model, threads=threads, batch_size=config.embed_batch_size)
    if provider in ("openai-compatible", "openai", "api"):
        base_url = emb.get("base_url")
        if not base_url:
            raise ValueError("embedding.provider is openai-compatible but embedding.base_url is not set")
        query_prefix = emb.get("query_prefix")
        if query_prefix is None and "qwen3-embedding" in model.lower():
            query_prefix = QWEN3_QUERY_PREFIX
        return OpenAICompatibleEmbedder(
            model=model, base_url=str(base_url),
            api_key_env=str(emb.get("api_key_env", "EMBEDDING_API_KEY")),
            dimensions=emb.get("dimensions"),
            batch_size=int(emb.get("batch_size", 64)),
            query_prefix=query_prefix or "",
            passage_prefix=str(emb.get("passage_prefix", "") or ""),
            timeout=float(emb.get("timeout", 60)),
            max_retries=int(emb.get("max_retries", 5)),
        )
    raise ValueError(f"unknown embedding.provider {provider!r}; use 'local' or 'openai-compatible'")
