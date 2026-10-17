## Spec

### Extensibility
- A source type may live outside this repo, named in config by its import path. [ADR 0005](docs/adr/0005-external-source-types-by-import-path.md)
  - Built-in types keep their short names.
- A source brings its own chunker.
- Chunks pass a transform stage before embedding, which may drop or rewrite them. [ADR 0006](docs/adr/0006-transform-stage-before-embedding.md)
  - Secret redaction is the first use.
  - Redaction masks, so the rest of the chunk stays searchable.

### Oracles
Algorithms and their tests are checked against oracle code before they are written. Clones live in `temple1-other/ref-repos`. Expiry for all: 2027-01-17; after it, check each is still active and useful.

#### Hybrid search
- tobi/qmd — FTS5 query building, weighted RRF, reranker blend.
- ariel-frischer/kb — same stack: vec0, FTS5, RRF, cross-encoder.
- qdrant/qdrant-client — fusion edge cases: empty lists, weights, zero variance.
- simonw/sqlite-utils — FTS5 query quoting.
- doobidoo/mcp-memory-service — RRF tests over sqlite-vec and FTS5.
- neuml/txtai — convex-combination fusion.
- raine/claude-history — RRF over Claude Code conversations.
- basicmachines-co/basic-memory — reranker pipeline tests. Read only, AGPL.

## Open questions

- Can search results be paged, when a reranker reorders them between calls?

## UNPROCESSED SPEC
(HUMANS: If you are a human, and any of this below yours, please deliberate above.)

- Airidas, approving hybrid search (FTS5 in the same store, OR queries, RRF over all lists, rerank the fused pool) and choosing its fusion: "RRF, k configurable, default 20"
- Airidas, on where hybrid is on by default: "Every instance"
- Airidas, on measuring it: "Add to roadmap to build a nice eval method and library. But for this one temporaruly setup en empheral little eval"
