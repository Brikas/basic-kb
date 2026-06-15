# basic-kb roadmap

Future capabilities — not built yet, captured so the design stays headed the right
way. Driven mostly by the LogSeq personal-KB use case, but each item should land as a
**general** engine feature, not a LogSeq special-case (configs/source-types carry the
specifics; the engine stays generic).

## Chunk filter / transform hook

A pluggable, per-source step that runs over each chunk (or its source text) before
embedding, able to **drop** a chunk or **rewrite** its text/attributes. Today a source
goes parse → chunk → embed with no place to clean things in between.

Motivating cases (LogSeq):
- Strip UI-only properties that carry no meaning — e.g. `collapsed:: true`. Pure editor
  state; it only pollutes embeddings.
- More generally: per-source line/attribute filters (strip patterns or a predicate).

Open design points: filter at parse time vs. a distinct post-chunk stage;
config-declared list of strip patterns (simple) vs. a code hook (richer logic).

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
way the `breadcrumb` chunker uses headings. Lands as a new source/chunker type once the
hook above exists.
