"""Finding a served instance from the CLI: `served.json` and the attach decision.

Liveness comes from the writer lock (lock.py), which the operating system releases on
any exit; `served.json` only answers "where" and carries a per-start nonce that
`GET /health` echoes. The decision procedure below therefore never has to guess whether
a file is stale:

1. `--attach URL` given: use it, skip everything local.
2. `--no-attach`: run locally (a write against a running server still fails with
   StoreBusy through the lock, with a message that says what to do).
3. Writer lock free: nobody is serving. A leftover `served.json` is stale by definition
   and is deleted. Run locally.
4. Lock held, no `served.json`: the writer is a plain `index`/`watch` or a server still
   starting. Run locally; nothing is deleted.
5. `served.json` from another hostname (carried over by a file sync): ignore it.
6. Probe `/health` (300 ms, one retry). Unreachable or busy: run locally, keep the file.
   Nonce mismatch: another process owns that port now; run locally, keep the file.
7. Served config path differs from ours, or `--model` asks for a different model: run
   locally.
8. Otherwise attach, sending `local.key` when the server has authentication on.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from .client import RemoteError, RemoteKnowledgeBase, resolve_api_key
from .config import Config
from .keys import LocalKey
from .lock import WriterLock

logger = logging.getLogger("basic_kb")

SERVED_FILENAME = "served.json"
NO_ATTACH_ENV = "BASIC_KB_NO_ATTACH"


@dataclass
class ServedInfo:
    """What a running server leaves in `<store_dir>/served.json`."""
    url: str
    nonce: str
    pid: int
    hostname: str
    model_id: str
    version: str
    config_path: str
    started_at: float
    auth: bool


def served_path(store_dir: Path) -> Path:
    return Path(store_dir) / SERVED_FILENAME


def write_served(store_dir: Path, info: ServedInfo) -> None:
    """Atomic: a reader never sees a torn file."""
    path = served_path(store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_served(store_dir: Path) -> Optional[ServedInfo]:
    """The file's content, or None when absent. A file that cannot be parsed is removed
    and logged: `os.replace` makes torn writes impossible, so garbage there is garbage."""
    path = served_path(store_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ServedInfo(**data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("removing unreadable %s: %s", path, e)
        remove_served(store_dir)
        return None


def remove_served(store_dir: Path) -> None:
    try:
        served_path(store_dir).unlink()
    except FileNotFoundError:
        pass


def attach(
    config: Config,
    *,
    no_attach: bool = False,
    attach_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_override: Optional[str] = None,
    probe_timeout: float = 0.3,
    on_note: Optional[Callable[[str], None]] = None,
) -> Optional[RemoteKnowledgeBase]:
    """A RemoteKnowledgeBase for this instance when one should be used, else None (run
    locally). `on_note` receives one line explaining a non-obvious outcome, for stderr."""
    def note(msg: str) -> None:
        logger.info("attach: %s", msg)
        if on_note is not None:
            on_note(msg)

    if attach_url:
        remote = RemoteKnowledgeBase(attach_url, api_key=resolve_api_key(api_key))
        remote.health(timeout=max(probe_timeout, 2.0))          # fail fast with a clear message
        return remote
    if no_attach or os.environ.get(NO_ATTACH_ENV, "").strip() not in ("", "0", "false", "no"):
        return None

    lock = WriterLock(config.store_dir)
    if lock.is_free():
        if served_path(config.store_dir).exists():
            note("removed a stale served.json (no server holds the store)")
            remove_served(config.store_dir)
        return None

    info = read_served(config.store_dir)
    if info is None:
        note("another process is writing this store (no served instance to attach to)")
        return None
    if info.hostname != socket.gethostname():
        note(f"ignoring served.json from another machine ({info.hostname})")
        return None
    if str(Path(info.config_path)) != str(config.path):
        note(f"served instance was started from {info.config_path}, this run uses {config.path}; running locally")
        return None
    if model_override and model_override != info.model_id:
        note(f"served instance embeds with {info.model_id!r}; --model {model_override!r} runs locally")
        return None

    key = resolve_api_key(api_key)
    if info.auth and not key:
        key = LocalKey(config.store_dir).read()
    remote = RemoteKnowledgeBase(info.url, api_key=key)
    health = None
    for attempt in range(2):
        try:
            health = remote.health(timeout=probe_timeout)
            break
        except RemoteError as e:
            last = e
            time.sleep(0.05)
    if health is None:
        note(f"served instance at {info.url} did not answer ({last}); running locally")
        return None
    if health.get("nonce") != info.nonce:
        note(f"the process at {info.url} is not the basic-kb server that wrote served.json; running locally")
        return None
    logger.info("attach: using the served instance at %s (pid %s)", info.url, info.pid)
    return remote
