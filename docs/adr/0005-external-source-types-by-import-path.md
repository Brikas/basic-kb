---
status: Accepted
date: 2026-10-14
authors: [Airidas Brikas]
assisting_agent: Claude Code, session 77610198-1f18-4f11-b4f6-5f7be52c3164
---

# 0005. Load source types from an import path named in config

## Context

Extending the engine means editing the engine. Source types live in a `SOURCE_TYPES` dict in `basic_kb/sources.py`, chunkers in a private `_CHUNKER_FACTORIES` dict, rerankers in `RERANKER_TYPES`, embedding models in a `SUPPORTED` map. None of them is dynamic; there is no `importlib` call in the package, and `pyproject.toml` declares only the CLI script.

The roadmap already states the opposite principle — "configs/source-types carry the specifics; the engine stays generic". The mechanism contradicts it, because a source type can only be generic by living in this repo.

The forcing case is ckb, a knowledge base over AI coding-agent conversation transcripts. Its parsing is larger than both built-in source types combined: a line-type allowlist, an overloaded line type disambiguated by field shape, a parent-pointer tree walk that prunes abandoned branches, blob and signature stripping, and dedup of turns replayed by resume and fork. More importantly, the format it parses changes on a vendor's undocumented schedule rather than ours.

Alternatives on the table:

- **A third built-in source type.** Rejected: it couples this engine's release train to a vendor's format, and puts fixtures for a format we do not control into our test suite.
- **Entry points via `importlib.metadata`.** Rejected: they only map a short name onto a path that config can already state. They require the external type to be pip-installed, and an edited entry-point table stays invisible until reinstall — painful while the source type is the thing being developed. Worth adding only when a second author needs discovery; [Datasette's rule of thumb](https://architecturenotes.co/p/datasette-simon-willison) is to add a hook once two plugins want it.
- **A file-format boundary**, where the consumer emits normalized segments and the engine indexes them through one generic segment source type. The strongest alternative, and rejected on cost: two on-disk representations of the same conversations, intermediates that go stale whenever the converter has not run, debugging spread across two repos, and no path to indexing a file while it is still being written. Its separation benefit survives the decision below anyway, since an import path keeps vendor-format churn entirely inside the consumer's repo.
- **Namespace packages, or a `contrib/` directory in this repo.** Rejected: contrib trees that share the core's release cycle have failed at scale. TensorFlow's reached 107 projects with ownership unenforced ([contrib sunset RFC](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)), and LangChain sunset `langchain-community` over a shared release cycle and uneven maintenance.

[Scrapy](https://docs.scrapy.org/en/latest/topics/addons.html) resolves components from dotted paths in settings, which is the precedent followed here.

## Decision

A source's `type:` value containing a colon is treated as an import path of the form `package.module:ClassName`. The engine imports it, verifies the result subclasses `DataSourceBase`, and names the path in any error it raises. Values without a colon keep resolving through the built-in registry, which stays the way built-ins are declared.

Chunkers need no equivalent mechanism. `DataSourceBase.make_chunker` is already an overridable method, so an externally loaded source supplies its own chunker object without the chunker registry being involved.

## Consequences

A source type can live in the repo that owns its subject matter, load from a loose file on `sys.path` with no install step, and iterate without reinstalling anything. Format churn in a vendor's file never reaches this repo.

The cost accepted is that the engine imports third-party code at runtime, so a broken external type surfaces as an import failure when config loads rather than as a validation message. `DataSourceBase` also becomes a de facto public interface: changing it breaks external types silently until they are next run. The failure mode to watch for is the one pytest has hit repeatedly, where a plugin needs hacks to work around a hook surface that is too thin, and ends up absorbed into core. If an external source type starts working around `DataSourceBase` rather than through it, widen the interface rather than move the type inside.
