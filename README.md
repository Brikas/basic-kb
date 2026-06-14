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
basic-kb status
basic-kb search --source list      # list configured sources
```

(`basic-kb …` and `python -m basic_kb …` are equivalent.)

**Config resolution**, in precedence order:
1. `--config PATH` — explicit override
2. `BASIC_KB_CONFIG=PATH` — set once per shell/instance
3. `basic-kb.yaml` found by walking up from the current directory

`--source` picks one (`--source notes`), several (`--source notes,docs`), or all
(default). Index is incremental; use `--force` to rebuild. `index --preview`
dry-runs the chunker without embedding.

Write queries as **statements**, not keywords — match the form of the text you're
searching. Full guide: [docs/rag-query-guide.md](docs/rag-query-guide.md).

## Config

Copy [config.example.yaml](config.example.yaml) into your instance folder as
`basic-kb.yaml` (the name auto-discovery looks for). All relative paths anchor to
the config file's own directory. Sources have a `type` (`markdown` or
`transcript`) and a `chunker` (`recursive` or `breadcrumb`).

## Secrets

The engine reads `JINA_API_KEY` from the environment (reranking is skipped if
absent). Provide it three ways, in precedence order: shell env > `--env-file PATH`
> `env_file:` in the config. Secrets never go in the config — only a path to a
dotenv file.

## Develop

Architecture and gotchas: [docs/developing.md](docs/developing.md). Modules:
`models`, `embedders`, `rerankers`, `textsplit`, `chunkers`, `sources`, `core`, `cli`.
