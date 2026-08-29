# basic-kb

A small, config-driven semantic search engine over markdown/text. Local embeddings
(FastEmbed/ONNX, no API) into a single SQLite file via sqlite-vec, exact cosine search, optional Jina reranking.

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

## Embeddings: local or API

Default is local FastEmbed/ONNX (`embedding_model: bge-small-en-v1.5`, nothing leaves the machine). Any OpenAI-compatible `/v1/embeddings` endpoint works too — DeepInfra, Nebius, OpenRouter, OpenAI, a self-hosted vLLM — via an `embedding:` block (`provider: openai-compatible`, `base_url`, `api_key_env`, optional `dimensions`, `batch_size`, `query_prefix`/`passage_prefix`); see [basic-kb.example.yaml](basic-kb.example.yaml). Keys come from the environment or the `env_file`, never the config. The model id stored with the index includes the provider model and requested dimensions, so changing either is refused until `index --switch-model`. Trade-off to make consciously: every chunk of your sources is sent to that provider.

## Store

Everything an instance indexes lives in one file, `<store_dir>/kb.sqlite3`: vectors (a sqlite-vec `vec0` table), chunk text and metadata, and the per-file manifest. Search is an exact cosine scan — no approximate index, so results are deterministic, a delete is a delete, and nothing has to be held in RAM; at the sizes this engine targets (thousands to low hundreds of thousands of chunks) a scan is milliseconds. Why not HNSW: [docs/adr/0001](docs/adr/0001-sqlite-vec-as-vector-store.md).

SQLite frees pages on delete but never shrinks the file by itself, so the store auto-`VACUUM`s after a write once the rows deleted since the last vacuum reach `vacuum.deleted_fraction` of live rows (default 0.2) and at least `vacuum.min_deleted` (default 1000). This runs inside the store, so `index`, `watch` and library callers all get it; `basic-kb vacuum` forces one. Switching embedding models is explicit: a mismatch between the config and the store refuses to index anything until you run `index --switch-model`, which wipes the store and re-embeds every source (the vector table's dimension is fixed, and mixing models is never allowed).

## Decisions

Load-bearing architectural choices and the alternatives they rejected are recorded in [docs/adr/](docs/adr/README.md).

## Auto-reindex (`watch`)

`basic-kb watch` is a foreground process that watches every enabled source and
re-embeds a file once it has been quiet for its debounce window — so the index
tracks your edits without you running `index` by hand. Ctrl-C to stop.

```bash
basic-kb watch                 # all sources, config debounce (default 30s)
basic-kb watch --debounce 300  # reindex 5 min after a file goes quiet (0 = immediately)
```

Indexing is incremental at two levels: a file whose hash is unchanged is skipped outright, and a changed file re-embeds only the chunks whose text changed — chunk ids are content hashes, so appending to a transcript costs one or two embeddings, not the whole file. Watch is configured **per source**: add a `watch:` block. It reacts only to content changes (reads are ignored) and honours the same `throttle:` config and `--throttle/--cores-fraction/--priority` flags as `index`. Peak RAM is bounded by `embed_batch_size` (default 8 texts per ONNX pass — onnxruntime never returns a batch's peak, so a long-lived watcher keeps whatever its biggest batch needed; 256 cost ~4 GB on a single 241-chunk file).

## Freshness nudges

`scan` diffs the files on disk against the index (using the per-file hashes recorded at index
time) and reports new/changed/deleted counts without embedding anything. After a
search, basic-kb nudges you about a source only once it has stayed stale for
`stale_after_days` (default 3), then at most once per `remind_every_days` (default 1)
until you re-index — re-indexing resets the clock. Tune or disable it under
`freshness:` in the config.

## Query writing

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

Copy [basic-kb.example.yaml](basic-kb.example.yaml) into your instance folder as
`basic-kb.yaml` (the name auto-discovery looks for). All relative paths anchor to
the config file's own directory. Sources have a `type` (`markdown` or
`transcript`) and a `chunker` (`recursive` or `breadcrumb`).

Default search parameters live under a `search:` block — `n` (result count),
`separate` (batch vs fused), `max_chars`, `content_type`, and `timing`. Each is
overridden per-run by its CLI flag (`--n`, `--separate`/`--fused`, `--max-chars`,
`--content-type`, `--timing`). Precedence: CLI flag > `search:` > built-in default.

A source can list `exclude:` patterns (gitignore-lite) to skip files under its
`path`: an unanchored name (`*.tmp`, `TODO*`) matches at any depth, a trailing
slash (`drafts/`) matches a folder and its contents, a pattern with a slash
(`archive/old*`) is anchored at the source root, and a leading `!` re-includes.
Excludes are honoured everywhere — index, scan, status, and live `watch`.

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

`env_file:` is anchored to the config's directory. `env_file_search_up: N`
(default 5) climbs up to N parent directories, nearest first, for the closest
dotenv when it isn't beside the config — it looks for the basename of `env_file`
(or `.env`), goes straight up only, and never enters sibling dirs. Handy for a
repo-root `.env` shared by several instances. Set it to 0 to disable the walk-up.

## Develop

Architecture and gotchas: [docs/developing.md](docs/developing.md). Modules:
`models`, `embedders`, `rerankers`, `textsplit`, `chunkers`, `sources`, `store`, `core`,
`watcher`, `cli`.

Planned capabilities (chunk filter/transform hook, reference expansion, LogSeq-aware
chunker): [ROADMAP.md](ROADMAP.md).
