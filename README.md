# basic-kb

A small, config-driven semantic search engine over markdown/text. Local embeddings
(FastEmbed/ONNX, no API) into ChromaDB, with optional Jina reranking.

One engine, many **instances**. An instance is just a config file pointing at a
store dir and a list of sources — so a work KB and a personal KB stay fully
separate, while sources within an instance can be queried alone or merged.

## Install

```bash
pip install -e .            # or: pip install -e /path/to/basic-kb
```

## Use

Put a `basic-kb.yaml` in your instance folder, then just run the commands from
inside it — the config is found automatically (like `git` finds `.git`):

```bash
cd my-instance/            # the folder holding basic-kb.yaml
basic-kb index             # build/refresh the index
basic-kb search "the thing I'm looking for"
basic-kb status            # chunk/doc counts
basic-kb scan              # what's new/changed/deleted vs the index (no embedding)
basic-kb search --source list      # list configured sources
```

(`basic-kb …` and `python -m basic_kb …` are equivalent.)

**Config resolution**, in precedence order:
1. `--config PATH` — explicit override
2. `BASIC_KB_CONFIG=PATH` — set once per shell/instance
3. `basic-kb.yaml` found by walking up from the current directory

`--source` picks one (`--source notes`), several (`--source notes,docs`), or all
(default). Index is incremental; use `--force` to rebuild. `index --preview`
dry-runs the chunker without embedding. A pre-scan hashes every file first, so
progress reads `[k/work]` (files actually being embedded), not `[i/all-files]`.

**Corruption guard.** If a large fraction (default 90%) of already-indexed files
change or disappear at once — usually a moved, corrupted, or re-pointed source —
an incremental index confirms before re-embedding: it prompts on a terminal and
aborts unattended, leaving the index untouched. `--yes` accepts, `--no-reindex-guard`
skips it for one run, `--reindex-threshold F` tunes the trip point. Configure under
`reindex_guard:` (`enabled`, `threshold`).

## Auto-reindex (`watch`)

`basic-kb watch` is a foreground process that watches every enabled source and
re-embeds a file once it has been quiet for its debounce window — so the index
tracks your edits without you running `index` by hand. Ctrl-C to stop.

```bash
basic-kb watch                 # all sources, config debounce (default 30s)
basic-kb watch --debounce 300  # reindex 5 min after a file goes quiet (0 = immediately)
```

Watch is configured **per source**: add a `watch:` block.

`scan` diffs the files on disk against the index (using a manifest written at index
time) and reports new/changed/deleted counts without embedding anything. After a
search, basic-kb runs this check every few days and prints a one-line nudge if a
source has drifted — tune or disable it under `freshness:` in the config.

Write queries as **statements**, not keywords — match the form of the text you're
searching. Full guide lives as a skill in consuming workspaces, e.g.
`.github/skills/rag-query-writing/SKILL.md` in temple1.

### One search vs. many: fused vs. batch

Two ways to pass multiple queries — they answer different needs:

- **Fused (default).** `search "price too high" "budget concern"` runs every query,
  pools the hits into **one** ranked list, and returns `--n` results total. The queries
  are treated as re-framings of a *single* information need, so they reinforce each other
  (higher recall for the same notes). Use it for HyDE / multi-angle phrasing of one thing.
- **Batch** (`--separate`, alias `--batch`). `search "coffee gear" "tax deadlines" --separate`
  runs each query **independently** and prints its own block of `--n` results — no merging.
  Use it when the queries ask for *different* things and you want a full result set for each,
  in one call. `--n` is then per query (2 queries × `--n 15` = up to 30 results).

Rule of thumb: same target, different words → **fused**; different targets → **batch**.

## Config

Copy [config.example.yaml](config.example.yaml) into your instance folder as
`basic-kb.yaml` (the name auto-discovery looks for). All relative paths anchor to
the config file's own directory. Sources have a `type` (`markdown` or
`transcript`) and a `chunker` (`recursive` or `breadcrumb`).

## Local overrides

Place a `basic-kb.local.yaml` (or `.yml`) next to your `basic-kb.yaml`. It's
deep-merged on top of the base config at load time — scalar and dict values
override, lists replace entirely.

Useful for machine-specific paths (e.g. a symlink that resolves differently per OS),
without touching the committed config. Add `**/basic-kb.local.yaml` to your
`.gitignore`.

## Secrets

The engine reads `JINA_API_KEY` from the environment (reranking is skipped if
absent). Provide it three ways, in precedence order: shell env > `--env-file PATH`
> `env_file:` in the config. Secrets never go in the config — only a path to a
dotenv file.

## Develop

Architecture and gotchas: [docs/developing.md](docs/developing.md). Modules:
`models`, `embedders`, `rerankers`, `textsplit`, `chunkers`, `sources`, `core`,
`watcher`, `cli`.

Planned capabilities (chunk filter/transform hook, reference expansion, LogSeq-aware
chunker): [ROADMAP.md](ROADMAP.md).
