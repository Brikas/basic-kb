# basic-kb roadmap

Future capabilities — not built yet, captured so the design stays headed the right
way. Driven mostly by the LogSeq personal-KB use case, but each item should land as a
**general** engine feature, not a LogSeq special-case (configs/source-types carry the
specifics; the engine stays generic).

## List
- Evaluation
  - Build a nice eval method and library: labelled queries per instance, metrics per retrieval mode, reranker and model, so changes are measured instead of judged by feel.
- Query Expansion
  - inspired by qmd: fine-tuned local 1.7B model emits lex:/vec:/hyde: sub-queries.
    - Custom fine-tuned tobil/qmd-query-expansion-1.7B (from Qwen3-1.7B), own training pipeline in finetune/
- Fusion tuning, once the eval exists
  - qmd's top-rank bonus (+0.05/+0.02) and rank-band blend of fusion vs reranker score (75/60/40%); both unablated upstream.
- Trigram keyword index, opt-in
  - Finds camelCase and word fragments; measured at ~6.5× the unicode61 index and ~7× slower.
- Deamon
  - Models stay resident, auto-unload after 5min idle (like qmd)

See temple1\notes\research\qmd-codebase-analysis.md

## LogSeq property stripping

A transform on the per-source transform stage ([ADR 0006](docs/adr/0006-transform-stage-before-embedding.md)) that strips UI-only properties carrying no meaning — e.g. `collapsed:: true`. Pure editor state; it only pollutes embeddings. Config-declared strip patterns cover it.

## Reference / link expansion

LogSeq notes point at other content — block refs `((id))`, embeds, `[[page]]` links,
blocks carrying an `id::`. A bare chunk loses whatever those refs point to.

Idea: expand referenced content so the meaning is present in results. **Open question —
where to do it:**
- at **chunk/embed time** (embed the expanded text), or
- only at **output time** (keep the index lean; resolve refs when showing a hit).

Constraints when built:
- Do **not** expand transclusion/embed blocks during embedding — fan-out can explode.
- Cap recursion depth; guard against reference cycles.

Current leaning (undecided): output-time expansion by default, with embed-time as an
opt-in for plain refs only.

## LogSeq-aware chunker

A breadcrumb-style chunker that uses the bullet **hierarchy** (nesting) as context, the
way the `breadcrumb` chunker uses headings. Lands as a new source/chunker type.
