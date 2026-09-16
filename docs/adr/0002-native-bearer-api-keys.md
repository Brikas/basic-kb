---
status: Accepted
date: 2026-09-11
authors: [Airidas Brikas]
assisting_agent: Claude (Fable 5.1), Claude Code session 6112e168-ce21-4e13-a74a-b33083d2b446
---

# 0002. Native bearer API keys for the served API, off by default, flat, wrapper-friendly

## Context

`basic-kb serve` exposes everything the store holds, and `POST /index` writes to it. The first remote consumer runs on another host. Three questions, settled together: does basic-kb authenticate at all, how much, and how does the local CLI avoid typing a credential on every command.

Forces:

- The engine is small and dependency-light (ADR 0001). A user-and-permission system is out of proportion.
- Others embed basic-kb behind a reverse proxy, a tailnet, a tunnel or their own identity layer. None of that should fight a built-in scheme.
- Anything that can read `store_dir` can already read `kb.sqlite3`. File permissions there are the existing local boundary.
- One consumer today, several tomorrow. A shared secret means one leak rotates everyone.

Alternatives considered:

- **No auth; deployments wrap it.** Each invents its own header, so the client and `--attach URL` cannot support them uniformly, and a stray `--host 0.0.0.0` serves the KB to the network. Rejected as the default, kept as an opt-out.
- **One static token in an env var.** No way to cut off one consumer without cutting off all. Rejected for a list of keys, which costs one file.
- **Exempt loopback from auth.** Moves the boundary from "can read `store_dir`" to "any process on this machine". Rejected for a key file with the same permissions as the database.
- **Users, roles, scopes, OAuth.** Out of proportion. Deferred; the key record has room.
- **Key management over the API.** A key that mints keys is privilege escalation without a permission model. Rejected; management stays local.

## Decision

A native, minimal bearer-key scheme.

- **Flat keys, several at once.** `bkb_` plus 32 URL-safe random characters. `<store_dir>/api-keys.json` (owner-only) stores a SHA-256 hash, id, name, prefix and timestamps. No scopes: every active key can do everything.
- **Mint and revoke locally**, never over the API: `basic-kb keys create|list|revoke`. A running server picks up a revocation without a restart.
- **Off by default.** `serve.auth: true` or `--auth` requires `Authorization: Bearer <key>` on every route, `/health` included, compared in constant time.
- **Guard rail.** A non-loopback bind with auth off refuses to start unless `--allow-unauthenticated` says something in front handles access.
- **Local attach needs no key.** With auth on, each start writes an ephemeral `<store_dir>/local.key` (owner-only) that a same-machine CLI reads. The boundary stays where it already was.
- **Remote credentials** come from `--api-key`, `BASIC_KB_API_KEY` or the config's `env_file`, like every other provider key.

## Consequences

Easier:

- One consumer's key rotates or dies without touching the others.
- A misconfigured exposure fails at startup instead of serving quietly.
- Wrappers disable auth with one config line and put anything in front; a bearer header survives every proxy for those who keep it on.
- One credential scheme for the client and the CLI to support.

Accepted costs:

- Every key is an administrator, `POST /index` included. A read-only consumer gets write access.
- TLS is out of scope: a key over plain HTTP is exposed, and basic-kb does not enforce transport security.
- Local security equals `store_dir` permissions. On a shared machine, whoever reads the key reads the database anyway.
- Minting a key for a remote consumer needs a shell on the serving machine.
- One more file format, and a `keys` command the parity test exempts deliberately.
