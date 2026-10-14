---
status: Accepted
date: 2026-10-14
authors: [Airidas Brikas]
assisting_agent: Claude Code, session 77610198-1f18-4f11-b4f6-5f7be52c3164
---

# 0006. A transform stage between chunking and embedding

## Context

A source goes parse, then chunk, then embed, with nowhere to clean anything in between. The roadmap has carried a "chunk filter / transform hook" as an open item for a while, motivated by LogSeq notes carrying editor state such as `collapsed:: true` that pollutes embeddings without meaning anything.

The ckb case turns that from cosmetic into load-bearing. Agent conversation transcripts contain raw tool output, which routinely includes live API keys, `.env` contents and bearer tokens. Once text is embedded, the secret is present in the vector whether or not the stored display text is scrubbed afterwards, so the cleaning has to happen before the embedder sees the text.

Measurements against a planted corpus of 720 secrets across 18 kinds, mixed with 5,280 lookalikes such as hashes, UUIDs and base64 blobs:

- Scanners driven by provider patterns miss key-name-shaped values. 80 of 720 planted secrets, of the `DB_PASSWORD="…"` form, were found only by rules keyed on the name rather than the value.
- Scanning raw JSONL instead of decoded text leaked a further 40, because escaped quotes break the patterns. The stage therefore has to run after parsing, on decoded text.
- A detector leaning on entropy alone is unusable without filters: one such tool leaked nothing but produced 504 false positives on lookalikes and 239 inside ordinary source code.

Alternatives on the table:

- **Filter inside each source's parse step.** Rejected: every source would re-implement it, and the property wanted — that nothing unredacted reaches the embedder — is a property of the engine, not of any one source.
- **Config-declared strip patterns only, with no code path.** Rejected: patterns cover line filters and redaction rules, but not predicates that must inspect structure. A configuration language that grows control flow becomes a bad programming language.
- **Leave it to each consumer to clean files before the engine sees them.** Rejected: it makes the guarantee opt-in per consumer, and the engine's own LogSeq case needs the same stage anyway.

This also settles the open design point the roadmap recorded, of filtering at parse time versus a distinct later stage, in favour of the later stage.

## Decision

A per-source transform stage runs over each chunk after chunking and before embedding. It may drop a chunk entirely, or rewrite its text and attributes. Simple cases are declared in config; a code hook covers logic configuration cannot express.

Secret redaction is the first implementation, and it masks rather than drops, so a redacted chunk stays searchable for everything around the secret.

## Consequences

Cleaning becomes a property of the engine rather than something each consumer has to remember, and the LogSeq property-stripping case lands in the same mechanism rather than needing its own.

The costs accepted are two. Indexing gets slower by whatever the transform does, on every chunk, on every run. More awkwardly, a transform that rewrites text changes chunk content, and chunk ids are derived from content — so enabling a transform, or changing one, invalidates every chunk in that source and forces a full re-embed. That is the correct behaviour, since stale chunks would otherwise keep an unredacted vector alive, but it means transform changes are expensive and should be batched rather than tweaked casually.
