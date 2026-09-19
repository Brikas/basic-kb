---
status: Accepted
date: 2026-09-19
authors: [Airidas Brikas]
assisting_agent: Claude (Fable 5.1 / Opus 5), Claude Code session 6112e168-ce21-4e13-a74a-b33083d2b446
amends: 0003
---

# 0004. Serve from a container, and replace discovery with a configured attach

## Context

ADR 0003 gave a served instance two mechanisms: an OS writer lock for mutual exclusion, and a `served.json` address file with a per-start nonce so a CLI on the same machine could find the server with no configuration. The lock has held up. The discovery half has not.

- Exposing the instance to the network meant putting it where everything else public on the host already lives: a container behind the existing reverse proxy. That is the box's standard, and it also isolates the service from the shared workspace virtualenv that any other tool can change underneath it.
- A container breaks discovery by construction. `served.json` records the writer's hostname, and the attach check ignored a file from a different host — a rule that exists to ignore a file carried in by a file sync, but which a container trips every time. The recorded URL is equally wrong: the container's loopback is not the host's.
- Weighed against that, discovery bought exactly one thing: not having to write an address in a config. Everything else it carried — the stale-file handling, the lock probe, the nonce, the hostname check — existed only to make that one convenience safe.
- A remote caller never had discovery anyway and always needed a URL. Keeping a second, different mechanism for the local case meant two code paths for one question.

Alternatives considered:

- **Keep discovery and teach it about containers** with an `advertise_url` setting and a relaxed hostname check. Works, but adds configuration to the mechanism whose only purpose was avoiding configuration.
- **Keep the host systemd service and route to it with a reverse-proxy file provider.** Fewer moving parts today, but it is throwaway work the moment the service is containerised, and it leaves the shared-virtualenv fragility in place.

## Decision

**Serve from a container.** One image, built without the `local` extra so an instance that embeds through an API carries no on-device model. Sources and the store are bind-mounted at the same absolute paths the host uses, so one config works in both places. The container runs as the store's owner, because the SQLite file and the writer lock are shared with any CLI on the host.

**Replace discovery with a configured attach.** An instance points at a served one with an `attach:` block (`url`, and `key_env` naming where the key lives — never the key itself). `--attach URL` overrides it. With no URL, the work happens locally. There is no address file and nothing to go stale.

**A URL is a promise.** If one is configured or given and the server cannot be used — unreachable, unauthorised, or outside the version floor — the command fails. It does not fall back to the local store, because silently answering from a different copy of the data is worse than not answering.

**The writer lock stays exactly as it was.** It is what stops two processes writing one store, and it works unchanged across the container boundary because a bind mount is the same inode.

**The local-key convenience survives.** A server with authentication on still writes an ephemeral key beside the store (ADR 0002), and a CLI that can read the store still picks it up, so nothing is typed on the box.

## Consequences

Easier:

- One mechanism answers "where does this run", for local and remote callers alike.
- `attach.py` is a short function instead of an eight-step procedure with a stale-state branch.
- The service no longer shares a virtualenv with every other tool in the workspace.
- A machine that only queries needs a four-line config and nothing installed beyond the package.

Accepted costs:

- Zero-configuration local attach is gone. A machine with a served instance needs one `attach:` block.
- A command fails when the server is down, where before it would have run locally. That is the intended trade, and `status` now names which source it read from so the two are never confused.
- Bind mounts must be correct or the watcher watches nothing. It is not silent — the watcher reports a missing directory and `status` prints a missing source path — but it is a new way to misconfigure.
- The container's uid must match the store's owner.
- Two installs can now drift, since the server ships in an image and the CLI is installed separately. `/health` advertises a minimum client version and the client refuses a pair outside the floor.
