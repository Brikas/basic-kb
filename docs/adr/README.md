# Architecture Decision Records

Small, numbered, append-only records of load-bearing choices — Context, Decision, Consequences — in the form of [Nygard (2011)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions). A decided record is never rewritten: reverse it with a new record and flip the old status to `Superseded by NNNN`. Skeleton: [template.md](template.md).

| # | Title | Status | Date |
|---|---|---|---|
| [0001](0001-sqlite-vec-as-vector-store.md) | Use sqlite-vec with exact search as the vector store, replacing Chroma/HNSW | Accepted | 2026-08-29 |
| [0002](0002-native-bearer-api-keys.md) | Native bearer API keys for the served API, off by default, flat, wrapper-friendly | Accepted | 2026-09-11 |
| [0003](0003-serve-attach-writer-lock.md) | Serve the engine over HTTP, attach the CLI to it, and gate writers with an OS lock | Accepted | 2026-09-11 |
