# Developing basic-kb

Read this before expanding or debugging the engine.

This engine was extracted from a single `knowledge.py` and split into the modules
listed below. It is config-driven: an *instance* is a YAML config naming a store
dir, an embedding model, chunker defaults, and a list of sources. Nothing about
any specific dataset lives in the code.

**If you change anything — add a gotcha, update the architecture, or fix a bug — update this file before closing the task.**

---

## Architecture

```
KnowledgeBase
├── EmbedderBase (ABC)
│   └── FastEmbedEmbedder   — ONNX-backed local embeddings via fastembed
├── RerankerBase (ABC)         — pluggable via RERANKER_TYPES; `reranker:` config picks one
│   ├── FastEmbedReranker   — local ONNX cross-encoder (no API key)
│   └── JinaReranker        — Jina AI API (/v1/rerank)
└── SearchResult (dataclass) — carries doc, metadata, cosine score, rerank_score
```

**Flow — index:**
1. `build_source()` builds each configured source; `parse_file()` → ParsedDocument
2. The source's chunker splits the body into Chunks
3. `FastEmbedEmbedder.embed()` → ChromaDB collection via custom `_ChromaEF` wrapper
4. Store chunk IDs, docs, metadata; skip already-indexed IDs

**Flow — search:**
1. Embed each query via the same `FastEmbedEmbedder`
2. `col.query()` per query, merge results by chunk ID (keep highest cosine score)
3. Optional: send top-N candidates to `JinaReranker.rerank()` → reorder by cross-attention score
4. Print results with both cosine score and rerank score (when reranking)

---

## Key files

| File | Purpose |
|---|---|
| `basic_kb/models.py` | Dataclasses: SearchResult, ParsedDocument, Chunk |
| `basic_kb/embedders.py` | EmbedderBase, FastEmbedEmbedder, ChromaDB EF wrapper |
| `basic_kb/rerankers.py` | RerankerBase, FastEmbedReranker, JinaReranker, `build_reranker` |
| `basic_kb/textsplit.py` | Recursive character splitter (`split_text`) |
| `basic_kb/chunkers.py` | ChunkerBase, RecursiveChunker, BreadcrumbHeadingChunker, `build_chunker` |
| `basic_kb/sources.py` | DataSourceBase, MarkdownSource, TranscriptSource, `build_source` |
| `basic_kb/core.py` | KnowledgeBase — index / search / status |
| `basic_kb/config.py` | Instance config + dotenv (`env_file`) loading |
| `basic_kb/cli.py` | argparse, commands, result printing |
| `<instance>/.chroma/` | Persistent ChromaDB index, lives next to the config (gitignored) |

---

## ChromaDB integration details

ChromaDB (v1.5.9+) uses a `Protocol`-based `EmbeddingFunction` with `__init_subclass__` that wraps `__call__` with validation. The inner `_ChromaEF` class in `as_chroma_ef()` must:
- Define `__init__` (even as a no-op) — the base class `__init__` is a sentinel that always warns
- Define `__call__` returning native Python `list[list[float]]` (not numpy types)
- Have a stable `__class__.__name__` so ChromaDB can detect EF conflicts on collection re-open

Changing the embedding model requires `index --force` — ChromaDB detects mismatches via the class name and raises `ValueError: embedding function conflict`.

---

## Model support

Adding a new FastEmbed model: add an entry to `FastEmbedEmbedder.SUPPORTED` dict. The key is the CLI alias; value is the full HuggingFace model ID. Any FastEmbed-supported HF id also works directly via `--model` without registering. Validate speed on your hardware before committing to a slow model — indexing is CPU-bound (see the CPU gotcha below).

Adding a new reranker: subclass `RerankerBase`, implement `rerank()`, and register it in
`RERANKER_TYPES` (rerankers.py). It then becomes selectable via `reranker: <key>` in any config.

---

## Gotchas

### 2026-05-11 — FastEmbed returns `np.float32`, ChromaDB rejects it

`list(numpy_array)` produces `[np.float32(...), ...]` which ChromaDB's `normalize_embeddings` rejects with:
```
ValueError: Expected embeddings to be a list of floats or ints...
```
**Fix:** use `.tolist()` instead — it recursively converts numpy scalars to native Python types.

```python
# Wrong
return [list(v) for v in self._model.embed(texts)]

# Correct
return [v.tolist() for v in self._model.embed(texts)]
```

---

### 2026-05-11 — `super().__init__()` in ChromaDB EF subclass always warns

`chromadb.EmbeddingFunction.__init__` is a sentinel: it unconditionally emits a `DeprecationWarning` saying the subclass doesn't implement `__init__`. Calling `super().__init__()` from your subclass triggers that warning every time.

**Fix:** define `__init__` as a plain `pass` — do NOT call super.

```python
class _ChromaEF(chromadb.EmbeddingFunction):
    def __init__(self) -> None:
        pass  # do not call super().__init__()
```

---

### Paths are anchored to the config file, never to the code

There is no "workspace root" assumption. Every relative path in a config
(`store_dir`, each source `path`, `env_file`) is resolved against the **config
file's own directory** (`config.base_dir` in `config.py`). Absolute paths pass
through unchanged. This is what lets the same engine back instances that live
anywhere. Don't reintroduce `Path(__file__)`-relative data paths.

---

### 2026-05-11 — `urllib` gets 403 from Cloudflare on the Jina API

The Jina API endpoint (`api.jina.ai`) sits behind Cloudflare. Bare `urllib.request` sends a minimal User-Agent that Cloudflare blocks with HTTP 403 / error code 1010.

**Fix:** use `requests` (a declared dependency). It sends a richer default User-Agent that passes Cloudflare.

```python
# Wrong — gets blocked
urllib.request.urlopen(req, timeout=15)

# Correct
requests.post(url, headers={...}, json={...}, timeout=15)
```

---

### 2026-05-11 — FastEmbed ONNX runs CPU-only on M1 by default

FastEmbed uses ONNX Runtime without Metal/CoreML acceleration by default. Benchmark results on M1 (1500-char chunks):

| Model | Speed | Full index (451 files) |
|---|---|---|
| bge-small-en-v1.5 | 8.3 chunks/s | ~10 min |
| bge-base-en-v1.5 | 2.5 chunks/s | ~32 min |
| nomic-embed-text-v1.5 | 1.8 chunks/s | ~45 min |
| all-MiniLM-L6-v2 | 90 chunks/s | < 1 min |

`all-MiniLM` is 10× faster but MTEB 56 vs 62 for `bge-small`. Since indexing is one-time, `bge-small` is the default. Do not switch to `bge-base` or `nomic` without testing — they are impractically slow on CPU.

---

### Secrets: env var first, then `--env-file`, then config `env_file`

The engine reads `JINA_API_KEY` straight from `os.environ` — it has no built-in
knowledge of any `.env` location (that would couple it to one workspace). `cli.main()`
loads, in order: `--env-file` (if passed), then the config's `env_file` (if set),
both via `os.environ.setdefault` — so an already-set shell variable always wins.
If reranking silently doesn't happen, the key isn't reaching the environment;
check that order. Reranking is skipped (not an error) when the key is absent.
