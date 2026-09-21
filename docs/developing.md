# Developing basic-kb

Read this before expanding or debugging the engine.

The engine is config-driven: an *instance* is a YAML config naming a store dir, an embedding model, chunker defaults and a list of sources. Nothing about any specific dataset lives in the code.

**If you change anything, add a gotcha, update the architecture, or fix a bug, update this file before closing the task. Every feature ships with a test.**

---

## Tests

```bash
python -m pytest                 # the whole suite, offline, a few seconds
python -m pytest tests/test_core.py -k guard
python -m pytest -m model        # opt-in: loads a real FastEmbed model (downloads on first run)
```

The suite never downloads or loads a model and never touches the network. `tests/fakes.py` provides `FakeEmbedder` (deterministic bag-of-words vectors: texts sharing words score higher, so ranking tests mean something) and `FakeReranker`; the sqlite-vec store, the watcher (real watchdog observer on a temp dir) and the server (in-thread uvicorn on a free port, plus a real `basic-kb serve` subprocess for the SIGKILL case) are exercised for real. `tests/conftest.py` builds a complete instance folder in `tmp_path`. Markers `model` and `network` are excluded by default (`pyproject.toml`).

`tests/test_parity.py` asserts that every operation in `core.OPERATIONS` exists on `KnowledgeBase` and `RemoteKnowledgeBase` with compatible signatures, has an API route and has a CLI command. Adding an operation means adding it to all four or the suite fails. CI (`.github/workflows/tests.yml`) runs the suite on Linux, macOS and Windows with Python 3.10 and 3.13.

---

## Architecture

```
Config (config.py)  ──►  KnowledgeBase.from_config  ──►  KnowledgeBase (core.py)
                                                         ├── EmbedderBase        EMBEDDER_PROVIDERS registry (embedders.py)
                                                         │   ├── FastEmbedEmbedder            local ONNX
                                                         │   └── OpenAICompatibleEmbedder     any /v1/embeddings endpoint
                                                         ├── RerankerBase        RERANKER_TYPES registry (rerankers.py)
                                                         │   ├── FastEmbedReranker            local cross-encoder
                                                         │   └── RerankAPIBase → Jina-/DeepInfra-compatible protocols
                                                         ├── SqliteVecStore      one SQLite file: vec0 + chunks + manifest (store.py)
                                                         └── WriterLock          one writer per store, OS-enforced (lock.py)

DataSourceBase  SOURCE_TYPES registry (sources.py)  ──►  ChunkerBase  _CHUNKER_FACTORIES (chunkers.py, textsplit.py)

Surfaces over the same operations (core.OPERATIONS):
  cli.py + render.py      argparse, attach decision, human/JSON rendering
  server.py               FastAPI routes + KBServer (lock for life, local key, optional Watcher)
  client.py               RemoteKnowledgeBase: same methods over HTTP; errors rebuilt as the same types
  attach.py               the CLI's attach decision: flag, env var, config
  keys.py                 API keys (ADR 0002)
  freshness.py            stale-source nudges (state in store_dir)
  watcher.py              Watcher: start/stop, events to a callback, one reindex worker
  serialize.py            to_jsonable / from_dict for every result dataclass (models.py)
```

**Flow, index:** `resolve_sources` builds each configured source; every file is hashed and only files whose hash differs from the store's `files` table are processed; the source's chunker splits the body; chunk ids are `rel_path::sha1(text)[:16]` so `SqliteVecStore.sync_file` embeds only chunks whose text is new and keeps the rest; one commit per file. `index_many` runs several sources with a shared `--limit` budget, handles `--switch-model`, and clears freshness nudges.

**Flow, search:** embed each query once; `SqliteVecStore.knn` per source (exact cosine, optional `content_type`), merge by chunk id keeping the highest score, optionally rerank the top candidates, return `SearchResult`s.

**Flow, serve/attach:** `KBServer` takes the writer lock, binds the socket, mints a `local.key` and serves; with `--watch` a `Watcher` runs inside it. A CLI command runs `attach.attach()`, which takes the first URL it finds: `--attach`, `BASIC_KB_ATTACH_URL`, then `attach_cli.url`. With none it runs locally; with one it resolves a key, checks identity and version over `/health` and forwards every call, raising rather than falling back. `serve`, `watch` and `keys` own the store and never attach. ADR 0003 covers the lock, 0004 the attach design.

---

## Key files

| File | Purpose |
|---|---|
| `basic_kb/models.py` | Result dataclasses: SearchResult, IndexResult, ReindexResult, ScanResult, SourceStatus, SourceInfo, InstanceInfo, PreviewFile, VacuumResult; ParsedDocument, Chunk |
| `basic_kb/serialize.py` | `to_jsonable` (fields + properties) and `from_dict`; used by CLI `--json`, the server and the client |
| `basic_kb/errors.py` | Typed errors; an empty result always means "nothing matched" |
| `basic_kb/embedders.py` | EmbedderBase, FastEmbedEmbedder, OpenAICompatibleEmbedder, `EMBEDDER_PROVIDERS`, `build_embedder` |
| `basic_kb/rerankers.py` | RerankerBase, FastEmbedReranker, RerankAPIBase + protocol subclasses, `RERANKER_TYPES`, `build_reranker` |
| `basic_kb/textsplit.py` | Recursive character splitter (`split_text`) |
| `basic_kb/chunkers.py` | ChunkerBase, RecursiveChunker, BreadcrumbHeadingChunker, `build_chunker` |
| `basic_kb/sources.py` | DataSourceBase, MarkdownSource, TranscriptSource, `build_source`, `resolve_sources`, `path_excluded` |
| `basic_kb/store.py` | SqliteVecStore: vec0 + chunks + files tables, KNN, auto-VACUUM policy, model guard |
| `basic_kb/lock.py` | WriterLock over `filelock`; `StoreBusy` |
| `basic_kb/core.py` | KnowledgeBase: `from_config`, index / index_many / reindex_paths / search / search_grouped / status / scan / info / vacuum / preview; `OPERATIONS` |
| `basic_kb/freshness.py` | FreshnessSettings, FreshnessTracker (state file `freshness_state.json`) |
| `basic_kb/watcher.py` | Watcher, WatchEvent, WatchSettings, `resolve_settings` |
| `basic_kb/keys.py` | ApiKeyStore (`api-keys.json`), LocalKey (`local.key`) |
| `basic_kb/attach.py` | `attach()`, `resolve_attach_key()`, `BASIC_KB_ATTACH_URL` / `BASIC_KB_NO_ATTACH` |
| `basic_kb/client.py` | RemoteKnowledgeBase, RemoteError, `resolve_api_key` |
| `basic_kb/server.py` | FastAPI app (`create_app`), KBServer, request models, error mapping, `require_key` |
| `basic_kb/config.py` | Instance config, local override merge, dotenv resolution, `find_config` |
| `basic_kb/render.py` | Human renderers and `emit_json` |
| `basic_kb/cli.py` | argparse, commands, attach wiring |
| `basic_kb/__init__.py` | Public API, `basic_kb.open`, `basic_kb.connect` |
| `<store_dir>/` | `kb.sqlite3`, `writer.lock`, `local.key`, `api-keys.json`, `freshness_state.json` |
| `docs/adr/` | 0001 sqlite-vec + exact search; 0002 API keys; 0003 serve, attach, writer lock; 0004 container + configured attach |

---

## Rules that hold everywhere

- **The library never prints.** Progress, confirmation and events are callbacks (`on_progress`, `on_confirm`, `on_event`, `on_warning`); the CLI passes printers, the server passes loggers. `cli.py`, `render.py` and `watcher`'s CLI renderer are the only places with `print`.
- **Registries, not if-chains.** Sources, chunkers, embedder providers and rerankers are dicts keyed by the config value. Register a class and it is selectable from any config; the CLI's choices follow the registry.
- **Data in, data out.** Operations return dataclasses in `models.py`; derived values are properties so `to_jsonable` carries them and `from_dict` ignores them. No second model layer for the API.
- **Errors are typed and travel.** The server maps each error class to a status and returns the class name; the client rebuilds it. `MassChangeRefused` carries its numbers both ways.
- **Every write takes both locks**, the process RLock first and the WriterLock second (`core.py`). The server and the watcher hold the WriterLock for their lifetime; the lock is reentrant across threads (`thread_local=False`).
- **Paths anchor to the config file**, never to the code. Every relative path in a config resolves against `config.base_dir`.
- **Secrets never live in the config**, only in the environment or a dotenv file it points to.

---

## sqlite-vec store details

- Two-level incrementality: file hash (`files` table) decides whether a file is looked at; chunk hash (`chunks.chunk_id`) decides what gets embedded. The positional index of a chunk lives in metadata (`position`), so reordering never forces a re-embed.
- One `vec0` table for all sources: `source` is a partition key, `content_type` a filterable metadata column, `embedding FLOAT[d] distance_metric=cosine`; `rowid` equals `chunks.id`. Metadata columns reject NULL, so a missing content_type is stored as `''`.
- The vector table is created on first write with a fixed dimension. `model_id`/`dim` live in `meta`; `check_model()` refuses index and search with a different model. A model switch goes through `prepare_model_switch(sources, accept)`: refuse unless `accept` (CLI `--switch-model`, API `switch_model: true`), then `clear_all()`. `--force` alone never switches models.
- Every store method opens its own WAL connection and closes it; reads are thread-safe, writes are serialised by the KnowledgeBase.
- Auto-vacuum counts deleted rows in `meta.deleted_since_vacuum`; a `database is locked` during VACUUM is logged and retried on the next write.
- Extension loading needs `sqlite3.Connection.enable_load_extension`; a Python build without it gets a `StoreError` naming the fix.

## Adding things

- **Embedding provider:** a factory `(config, threads=None) -> EmbedderBase` in `EMBEDDER_PROVIDERS`.
- **Reranker:** subclass `RerankerBase` (or `RerankAPIBase` for bearer-auth HTTP) and register in `RERANKER_TYPES`.
- **Source type / chunker:** subclass and register in `SOURCE_TYPES` / `_CHUNKER_FACTORIES`.
- **Operation:** a `KnowledgeBase` method, the same method on `RemoteKnowledgeBase`, a route in `server.py`, a CLI command or flag, and its name in `core.OPERATIONS`; then tests for each surface.

---

## Gotchas

### A watcher that reacts to inotify read events loops on itself

watchdog ≥ 4 on Linux emits `opened`/`closed_no_write` for plain reads. The watcher's own reindex reads the file, so an unfiltered handler requeues every file it just processed and loops at the debounce period. `_Handler.dispatch` ignores those event types and `reindex_paths` skips unchanged hashes. Both are covered by tests; do not remove either.

### fastembed's default `batch_size=256` peaks at ~4 GB on one 241-chunk file

Attention memory grows with batch × seq_len² and onnxruntime's arena never returns the peak. `embed_batch_size` (default 8) bounds it at ~0.5 GB with identical throughput.

### An emptied source must still reconcile, or its vectors are orphaned

`index()` reads the manifest before the empty-files check, so a source whose directory emptied prunes every file instead of returning early. `--limit` runs never prune (they see a subset), and a never-indexed source with no `--force` is a no-op. The mass-change guard covers the emptied case too: 5+ indexed files vanishing is 100% churn and an unattended run refuses. Covered by tests.

### A dropped WriterLock object releases the lock

`WriterLock(dir).acquire()` without keeping the object releases as soon as it is garbage collected. Long-lived holders keep the lock on an object that lives as long as they do; the KnowledgeBase owns one.

### Key-file cache invalidation keys on the inode

Linux mtime granularity is coarse enough for two writes to share a timestamp. `ApiKeyStore` re-reads when `(inode, mtime_ns, size)` changes; every save replaces the file, so the inode changes.

### `urllib` gets 403 from Cloudflare on the Jina API

Bare `urllib.request` sends a minimal User-Agent that Cloudflare blocks. Use `requests` (a declared dependency).

### FastEmbed ONNX runs CPU-only by default

Benchmarks on M1 (1500-char chunks): bge-small-en-v1.5 8.3 chunks/s, bge-base 2.5, nomic 1.8, all-MiniLM-L6-v2 90. `bge-small` is the default; do not switch to `bge-base` or `nomic` without measuring.

### Secrets: env var first, then `--env-file`, then config `env_file`

The engine reads provider keys straight from `os.environ`. `cli.main()` and `basic_kb.open()` load the dotenv files with `setdefault`, so an already-set shell variable always wins. If reranking silently does not happen, the key is not reaching the environment.
