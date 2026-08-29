---
status: Accepted
date: 2026-08-29
authors: [Airidas Brikas]
assisting_agent: Claude (Fable 5), Claude Code session 1ae52599-d834-4c34-ad24-4f62b78266ca
---

# 0001. Use sqlite-vec with exact search as the vector store, replacing Chroma/HNSW

## Context

basic-kb is a personal knowledge base: today 7,094 chunks across three sources (chats 2,562 · logseq 4,014 · meetings 518), 384-dimensional `bge-small-en-v1.5` vectors, 10.4 MB of float32 in total. It must run in-process on macOS, Windows and Linux, with no server. The workload is update-heavy: a watcher re-embeds chat transcripts and notes as they change, every few minutes, around the clock.

The store since the first commit (2026-06-14) has been Chroma with its HNSW index. On 2026-08-27/29 this took the host down twice. Root causes, measured on the box:

- HNSW never reclaims deleted vectors. Chroma's local segment marks a label deleted and allocates every re-add a fresh slot; capacity is resized on allocated-including-deleted count ([local_hnsw.rs](https://github.com/chroma-core/chroma/blob/main/rust/segment/src/local_hnsw.rs)). The chats collection reached 341,646 allocated slots for 2,362 live chunks — a 572 MB index file for ~4 MB of vectors — and the watcher was OOM-killed at 5–5.7 GB RSS three times. The only remedy is a full collection rebuild.
- Chroma will not fix this: the feature request is open and unassigned since 2024 ([#2594](https://github.com/chroma-core/chroma/issues/2594)); the `allow_replace_deleted` PR was closed unmerged on 2026-07-15 ([#2621](https://github.com/chroma-core/chroma/pull/2621)); no knob is exposed in 1.5.9.
- The whole HNSW graph must live in RAM ([Chroma docs](https://docs.trychroma.com/guides/performance/single-node)); recall drifts as tombstones accumulate ([FreshDiskANN](https://arxiv.org/abs/2105.09613)); soft-deleted private-chat vectors remain recoverable from the index files until a rebuild ([Ghost Vectors](https://arxiv.org/abs/2606.18497)).
- Chroma pulls 28 direct dependencies (kubernetes, grpcio, opentelemetry, uvicorn, …) and ~64 MB of native code, and its on-disk format has changed across major versions.

What the engine actually needs from a store is small: one collection per source; `add`/`delete` by id; look up a file's chunks by `rel_path`; cosine top-k with an optional `content_type` equality filter; `count`; a metadata dump. Embedding is done by basic-kb itself.

The decisive measurement: an exact brute-force cosine scan over 384-dim float32 on the 4-vCPU host takes 0.33 ms at 5k vectors, 1.9 ms at 20k, 5.6 ms at 100k and 33 ms at 500k. At today's size an approximate index saves about one millisecond per query, in a pipeline whose query embedding and cloud rerank cost tens to hundreds of milliseconds. HNSW exists to make search sub-linear when a linear scan is too slow; here the scan is not slow.

Alternatives considered and rejected:

- **Keep Chroma and add a compaction job** (copy live vectors into a fresh collection when fragmentation exceeds ~20%). Works, but leaves the RAM-resident graph, the tombstones between compactions, the single-writer rule and a maintenance job that must run forever. Rejected as the minimum fix, not the right one.
- **LanceDB**: real compaction (`optimize()`), disk-native IVF-PQ, SQL filters, hybrid full-text search. Rejected for now: pyarrow + Rust wheels, v0.x API churn (filter-chaining semantics changed at 0.34), versioned storage that grows until cleanup — more machinery than a 10 MB corpus warrants. Remains the designated step-up if the corpus passes ~200k chunks or hybrid search is wanted.
- **Milvus Lite v3** (pure-Python rewrite, LSM compaction, HNSW/IVF per segment): the best delete story on paper, but weeks old, Windows support only via faiss-cpu/pyarrow wheels, runs a local gRPC adapter. Rejected on maturity.
- **libSQL/Turso DiskANN**: on-disk ANN native to SQLite, but the recommended Python package (`pyturso`) has no Windows wheel and libSQL itself has been declared legacy in favour of the Turso rewrite. Rejected.
- **DuckDB VSS**: HNSW in RAM with manual compaction; persistence documented as experimental and not for production. Rejected.
- **Qdrant local mode**: numpy brute force meant for prototyping; the client warns above 20k points. Rejected.
- **vectorlite, sqlite-vector**: stale beta (last release 2024-08) and a non-OSI license without a PyPI package respectively. Rejected.
- **Index libraries (FAISS Flat, usearch exact, plain numpy) over our own SQLite tables**: same properties as the chosen option with ~150 more lines of persistence and filtering code to own. Kept as the fallback if extension loading is unavailable on some Python build.

## Decision

Replace Chroma with **sqlite-vec** as the vector store, using exact (brute-force) k-nearest-neighbour search and no approximate index.

- Vectors live in a `vec0` virtual table keyed by chunk rowid, with `source` as a partition key and `content_type` as a metadata column so the KNN query can filter on them.
- Chunk text, `rel_path`, `chunk_hash` and other metadata live in ordinary SQLite tables in the same database file; the manifest moves into the same file. File-level lookups and deletes are plain SQL, never KNN.
- Vectors are stored as rows, not as an opaque index, so a later ANN layer (sqlite-vec's own IVF/DiskANN when released, or LanceDB) is a copy of existing rows, never a re-embed.
- The `KnowledgeBase` interface and CLI stay as they are; only the storage backend changes. Chunk-level content hashing (embedding only chunks whose text is new) lands in the same piece of work, since it removes most churn regardless of store.

## Consequences

Easier:

- A delete is a delete; there is no tombstone accumulation, no compaction job, no rebuild command to remember.
- Memory stays flat and small: the store streams vector pages from disk instead of holding a graph in RAM; the watcher stops being a multi-GB process.
- Results are exact and deterministic; no recall drift, no `ef`/`M` tuning.
- One file per KB, one dependency with zero transitive dependencies (~1 MB), wheels for all three platforms, SQL for every filter the engine needs. Backups are a file copy.
- Deleted private text is actually gone once `VACUUM` runs.

Accepted costs:

- Query time grows linearly with corpus size. Measured brute-force latency is negligible below ~100k chunks and ~33 ms at 500k on this hardware; sqlite-vec itself was not benchmarked here and an independent 2026 benchmark puts it roughly 10–100× slower than an in-RAM scan at 100k × 1024-dim. If the corpus grows past a few hundred thousand chunks or a sub-10 ms budget appears, an ANN index must be added — by copying rows, which this design keeps cheap.
- sqlite-vec is pre-1.0, effectively single-maintainer, with a main branch quiet for three months at the time of writing; the `DELETE` bug fixed in 0.1.9 is a reminder of that maturity level.
- The database file does not shrink on delete until `VACUUM` is run; the engine must schedule it.
- Requires `sqlite3.Connection.enable_load_extension`, which some Python builds (historically python.org macOS) omit. Verify on every target machine before rollout; the numpy-over-SQLite fallback covers a build without it.
- A one-time migration: rebuild the index (or copy embeddings out of the existing Chroma collections) and delete the `.chroma` directories.
- We give up Chroma's richer `where` DSL and polished API; sqlite-vec metadata filters support equality and comparison operators, up to 16 metadata columns, and text values over 12 characters are slower in KNN filters — hence `rel_path` stays in an ordinary table.
