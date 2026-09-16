---
status: Accepted
date: 2026-09-11
authors: [Airidas Brikas]
assisting_agent: Claude (Fable 5.1), Claude Code session 6112e168-ce21-4e13-a74a-b33083d2b446
---

# 0003. Serve the engine over HTTP, attach the CLI to it, and gate writers with an OS lock

## Context

Three surfaces have to stay on par: CLI, module, and a long-lived process other agents reach over the network. Before this record the CLI was the only complete one — the composition root, source resolution, freshness and preview all lived in `cli.py`, and there was no API.

Two long-lived writers already existed on the box: the watcher unit and whatever the user ran by hand. The in-process `RLock` serialises writers inside one process only; the README asked the user to coordinate by hand.

Attaching the CLI to a running server saves a model load of seconds and hundreds of megabytes, but needs a liveness signal that survives every way a process can die.

- A pid or state file goes stale on `kill -9`, an OOM kill or a power loss, and guessing "alive but slow" wrong puts two writers on one SQLite file. Heartbeats lie on a suspended or clock-skewed machine.
- An advisory OS lock has one owner or none at every instant on all three platforms, and the kernel releases it on every exit path. Verified on the box: a second process is refused while the first holds it, and the lock is free within a second of a SIGKILL. `filelock` wraps `fcntl.flock` and `msvcrt.locking` and already ships as a fastembed dependency.
- Machines hold many instances, so anything keyed on a fixed port or socket path collides; `store_dir` does not.

Alternatives considered:

- **A pid or state file alone.** Stale by construction; the failure mode is a write race.
- **A fixed port, or a unix domain socket.** Instances collide; unix sockets are second-class on Windows.
- **Keep `watch` and `serve` as separate processes.** Two copies of the model, still competing for writes.
- **A job queue for indexing.** One synchronous run under a lock and a 409 covers the real usage; a queue can come later without changing routes.
- **A second, pydantic model layer for the wire.** JSON over the existing dataclasses keeps one source of truth and lets the CLI render remote results with local code.

## Decision

**One writer per store, enforced by the OS.** Every write takes `<store_dir>/writer.lock`; `serve` and `watch` hold it for their lifetime. A second process fails with `StoreBusy`. Readers never take it.

**`basic-kb serve`.** FastAPI over one `KnowledgeBase`, routes mirroring the library, errors carrying the exception class so the client rebuilds the same types. `--watch` runs the watcher inside it. Loopback by default; auth is ADR 0002.

**`served.json` answers "where", the lock answers "whether".** The server writes URL, a per-start nonce echoed by `/health`, hostname, model and config path. A CLI with a free lock treats any file as stale and deletes it; with the lock held it probes `/health` and attaches only on a matching nonce, hostname, config and model. `--no-attach` skips it, `--attach URL` targets a remote server. Writes attach too, so the server stays the only writer.

**One composition root.** `KnowledgeBase.from_config` builds an instance; `basic_kb.open` and `basic_kb.connect` return local and remote objects with the same methods. A parity test asserts every operation has a method on both, a route and a CLI command.

## Consequences

Easier:

- A second writer is impossible rather than discouraged.
- A killed server leaves nothing that misleads the next command. Tested by SIGKILLing a real `serve` subprocess.
- The CLI on the serving machine costs no model load and no key entry.
- Module users get the engine from one call, local or remote, with the same dataclasses back.

Accepted costs:

- The store must be on a local disk; NFS and SMB have no reliable lock and this design does not detect that.
- `POST /index` is synchronous: a concurrent run is refused with 409 rather than queued.
- Attach adds a lock probe and one HTTP round trip (300 ms budget) per command.
- A `WriterLock` object that is not kept referenced releases on garbage collection.
- Between the lock and uvicorn listening there is a window where a probing CLI falls back to a local run for that one command.
