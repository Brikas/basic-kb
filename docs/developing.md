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
│   ├── FastEmbedEmbedder   — ONNX-backed local embeddings via fastembed (returns list[list[float]])
│   └── OpenAICompatibleEmbedder — any /v1/embeddings HTTP endpoint; batching, retries, prefixes; `build_embedder()` picks by config
├── SqliteVecStore          — one SQLite file: vec0 vectors + chunks + per-file manifest; auto-VACUUM
├── RerankerBase (ABC)         — pluggable via RERANKER_TYPES; `reranker:` config picks one
│   ├── FastEmbedReranker   — local ONNX cross-encoder (no API key)
│   └── JinaReranker        — Jina AI API (/v1/rerank)
└── SearchResult (dataclass) — carries doc, metadata, cosine score, rerank_score
```

**Flow — index:**
1. `build_source()` builds each configured source; `parse_file()` → ParsedDocument
2. Hash every file; only files whose hash differs from the store's `files` table are processed
3. The source's chunker splits the body into Chunks; `FastEmbedEmbedder.embed()` (batched, `embed_batch_size`)
4. Chunk ids are `rel_path::sha1(text)[:16]` (`#n` for repeated text). `SqliteVecStore.sync_file()` diffs them against the stored rows: unchanged chunks keep their vector (metadata refreshed), vanished ones are deleted, only new ones are embedded; one commit per file

**Flow — search:**
1. Embed each query once via the same `FastEmbedEmbedder`
2. `SqliteVecStore.knn()` per source (exact cosine, `k`, optional `content_type`), merge by chunk id (keep highest score)
3. Optional: send top-N candidates to `JinaReranker.rerank()` → reorder by cross-attention score
4. Print results with both cosine score and rerank score (when reranking)

---

## Key files

| File | Purpose |
|---|---|
| `basic_kb/models.py` | Dataclasses: SearchResult, ParsedDocument, Chunk |
| `basic_kb/embedders.py` | EmbedderBase, FastEmbedEmbedder (batch size bounds peak RAM) |
| `basic_kb/rerankers.py` | RerankerBase, FastEmbedReranker, JinaReranker, `build_reranker` |
| `basic_kb/textsplit.py` | Recursive character splitter (`split_text`) |
| `basic_kb/chunkers.py` | ChunkerBase, RecursiveChunker, BreadcrumbHeadingChunker, `build_chunker` |
| `basic_kb/sources.py` | DataSourceBase, MarkdownSource, TranscriptSource, `build_source` |
| `basic_kb/store.py` | SqliteVecStore — vec0 + chunks + files tables, KNN, auto-VACUUM policy, model guard |
| `basic_kb/core.py` | KnowledgeBase — index / reindex_paths / search / status / scan / info over the store |
| `<store_dir>/kb.sqlite3` | the whole index: vectors, chunk text, metadata, per-file hashes (`files` table powers `scan`) |
| `<store_dir>/freshness_state.json` | per-source last freshness-check timestamp |
| `basic_kb/config.py` | Instance config + dotenv (`env_file`) loading |
| `basic_kb/cli.py` | argparse, commands, result printing |
| `docs/adr/` | architecture decision records; 0001 = why sqlite-vec + exact search |

---

## sqlite-vec store details

- Two-level incrementality: file hash (`files` table) decides whether a file is looked at; chunk hash (`chunks.chunk_id`) decides what gets embedded. Positional index of a chunk lives in metadata (`position`), not in its id, so reordering never forces a re-embed.
- One `vec0` virtual table for all sources: `source` is a **partition key** (pre-filters the scan), `content_type` a **metadata column** (filterable in the KNN `WHERE`), `embedding FLOAT[d] distance_metric=cosine`. `rowid` equals `chunks.id`. Metadata columns are strictly typed and **reject NULL** — the store writes `''` for a missing content_type.
- The vector table is created on the first write and its dimension is fixed. `model_id`/`dim` live in `meta`; `check_model()` refuses index/search with a different model. A model switch goes through `KnowledgeBase.prepare_model_switch(sources, accept)` (called by `index` and `watch`): on a mismatch it refuses to index anything unless `accept` (CLI `--switch-model`, which implies `--force` and all sources), then empties the whole store first (`store.clear_all()`). `--force` alone never switches models. Per-source clearing cannot do this — the first source's write would meet the other sources' old-dimension vectors.
- Every store method opens its own connection (`PRAGMA journal_mode=WAL`), loads the extension, and closes — thread-safe reads; writers are serialised by `KnowledgeBase._write_lock` (process-local).
- Auto-vacuum: `meta.deleted_since_vacuum` counts deleted rows; `_maybe_vacuum()` runs after every write and VACUUMs when the policy (`VacuumPolicy`, from the `vacuum:` block) says so. A `database is locked` during VACUUM is logged and retried on the next write, never raised.
- Extension loading needs `sqlite3.Connection.enable_load_extension`; some Python builds lack it (historically python.org macOS) and get a `StoreError` naming the fix.
- TEMPORARY (2026-08-29): `legacy_chroma_leftovers()` + the CLI notice detect a pre-sqlite-vec `.chroma` store and tell the user to `index --force`. Delete both once all instances are migrated.

## Model support

Adding a new FastEmbed model: add an entry to `FastEmbedEmbedder.SUPPORTED` dict. The key is the CLI alias; value is the full HuggingFace model ID. Any FastEmbed-supported HF id also works directly via `--model` without registering. Validate speed on your hardware before committing to a slow model — indexing is CPU-bound (see the CPU gotcha below).

Adding a new reranker: subclass `RerankerBase`, implement `rerank()`, and register it in
`RERANKER_TYPES` (rerankers.py). It then becomes selectable via `reranker: <key>` in any config.

---

## Gotchas



### 2026-08-29 — a watcher that reacts to inotify read events loops on itself

watchdog ≥ 4 on Linux emits `opened`/`closed_no_write` for plain reads. The watcher's own reindex reads the file, so an unfiltered handler re-queued every file it had just processed and looped at the debounce period; Chroma's HNSW then grew 145× from the delete+add churn and the process was OOM-killed. `_Handler.dispatch` ignores those event types and `reindex_paths` skips unchanged hashes. Do not remove either.

---

### 2026-08-29 — fastembed's default `batch_size=256` peaks at ~4 GB on one 241-chunk file

Attention memory ∝ batch × seq_len² and onnxruntime's arena never returns the peak. `embed_batch_size` (default 8) bounds it at ~0.5 GB with identical throughput. Measured on bge-small, 2 threads.

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
