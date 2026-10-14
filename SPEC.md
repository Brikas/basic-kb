## Spec

### Extensibility
- A source type may live outside this repo, named in config by its import path. [ADR 0005](docs/adr/0005-external-source-types-by-import-path.md)
  - Built-in types keep their short names.
- A source brings its own chunker.
- Chunks pass a transform stage before embedding, which may drop or rewrite them. [ADR 0006](docs/adr/0006-transform-stage-before-embedding.md)
  - Secret redaction is the first use.
  - Redaction masks, so the rest of the chunk stays searchable.

## Open questions

- Can search results be paged, when a reranker reorders them between calls?
