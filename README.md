# basic-kb

A small, config-driven semantic search engine over markdown/text. Run locally (FastEmbed/ONNX) or through any OpenAI-compatible API, vectors live in a single SQLite file via sqlite-vec with exact cosine search. Supports reranking.

An **instance** is a config file: a store dir and a list of sources. Three ways to drive it, kept on par by a test: CLI, module, HTTP API.

## Install

```bash
pip install -e .              # CLI + module
pip install -e ".[serve]"     # + basic-kb serve
pip install -e ".[dev]"       # + tests
```

## CLI

Run from the folder holding `basic-kb.yaml` ([example](basic-kb.example.yaml)); it is found by walking up, the way `git` finds `.git`.

```bash
basic-kb index                  # incremental; --force --switch-model --limit N --preview
basic-kb search "a statement the note would contain"   # --n --offset --separate --source --content-type
basic-kb status                 # counts, staleness, index age
basic-kb scan                   # what changed since the last index
basic-kb info                   # what each source holds
basic-kb serve --watch          # HTTP API + watcher, one process
basic-kb watch                  # watcher alone
basic-kb vacuum
basic-kb keys create --name NAME
```

`--json` works on every command; progress, nudges and warnings go to stderr. `--source a,b`, `--source list`. Config override: `--config PATH` or `BASIC_KB_CONFIG`. `basic-kb` and `python -m basic_kb` are the same thing.

Write queries as statements, not keywords. Several queries fuse into one ranked list; `--separate` gives each its own. `--offset N` pages further down that list, within the rerank candidate ceiling — reranking cost stays fixed however deep you go, and a page cut short by the ceiling says so on stderr.

## Module

```python
import basic_kb

kb = basic_kb.open("my-instance/basic-kb.yaml")     # local; open() alone discovers the config
kb = basic_kb.connect("http://box:8765", api_key)   # served, same methods

kb.search(sources, ["..."], n=10)
kb.index_many(sources)
kb.status(sources)
```

Returns dataclasses (`basic_kb.models`), raises typed errors (`basic_kb.errors`), prints nothing.

## Serve

```bash
basic-kb serve --watch --auth
```

`GET /health /info /status /scan`, `POST /search /index /vacuum /preview`.

A CLI run attaches when it has a URL — `--attach URL`, `BASIC_KB_ATTACH_URL`, or `attach_cli:` in the config — and reaches the server instead of loading models again. The env var is there so one committed config serves every machine: the box running the server points it at its own loopback, everyone else falls through to the configured host. `--no-attach` works on the store directly. One writer per store, enforced by an OS lock: [ADR 0003](docs/adr/0003-serve-attach-writer-lock.md), [ADR 0004](docs/adr/0004-container-and-configured-attach.md).

Auth is off by default ([ADR 0002](docs/adr/0002-native-bearer-api-keys.md)). With `--auth` every route needs `Authorization: Bearer <key>`:

```bash
basic-kb keys create --name laptop        # printed once
basic-kb keys list
basic-kb keys revoke <id>
```

The CLI on the serving machine needs no key. Elsewhere: `--api-key`, `BASIC_KB_API_KEY`, or `connect(url, key)`. TLS is the deployment's job.

## Config

[basic-kb.example.yaml](basic-kb.example.yaml) documents every key inline. Relative paths anchor to the config file. `basic-kb.local.yaml` deep-merges on top for machine-specific paths (gitignore it). Secrets live in the environment or a dotenv the config points at.

Switching embedding model or provider changes the vector space, so it is refused until `index --switch-model`. An API provider receives every chunk of your sources.

## Develop

```bash
python -m pytest              # offline, ~20s; CI runs Linux/macOS/Windows on 3.10 and 3.13
python -m pytest -m model     # opt-in: loads a real model
```

[docs/developing.md](docs/developing.md) · [ADRs](docs/adr/README.md) · [ROADMAP.md](ROADMAP.md)
