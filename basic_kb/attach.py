"""Deciding whether a command runs against a served instance or against the local store.

One rule, one path: a URL is given (`--attach URL`), set for the machine
(`BASIC_KB_ATTACH_URL`) or configured (`attach_cli.url`), in that order — or there is none
and the work happens locally. There is no discovery, no state file, and nothing to go stale.

A URL is a promise that the work happens there, so anything that stops it raises rather
than quietly falling back. A command that silently searched a stale local copy because the
server was down would be worse than one that failed.

Mutual exclusion between writers is a separate concern and stays on the OS lock in
`lock.py`. That is what stops two processes writing one store; it was never what told a
client where to connect.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from .client import RemoteKnowledgeBase, resolve_api_key
from .config import Config
from .errors import BasicKBError
from .keys import LocalKey

logger = logging.getLogger("basic_kb")

NO_ATTACH_ENV = "BASIC_KB_NO_ATTACH"

# Where this machine reaches the served instance, overriding `attach_cli.url`. One config
# committed for every machine then needs no per-machine copy: the box running the server
# points this at its own loopback, everyone else falls through to the configured URL.
ATTACH_URL_ENV = "BASIC_KB_ATTACH_URL"


def _env_says_no_attach() -> bool:
    return os.environ.get(NO_ATTACH_ENV, "").strip() not in ("", "0", "false", "no")


def resolve_attach_key(config: Config, api_key: Optional[str] = None) -> Optional[str]:
    """The key to present, in precedence order: an explicit one, the env var the config
    names, the file the config names, `BASIC_KB_API_KEY`, then the local key a server left
    beside this instance's own store (ADR 0002) so nothing is typed on the box.

    A config that names an env var or a file and finds neither stops the run. Calling
    anyway returns 401, which sends you looking at the server instead of at the shell that
    forgot to export it.
    """
    if api_key:
        return api_key
    if config.attach_key_env:
        key = os.environ.get(config.attach_key_env) or None
        if key:
            return key
        if not config.attach_key_file:
            raise BasicKBError(
                f"attach_cli.key_env names {config.attach_key_env}, but it is not set. Put the key in "
                f"the environment or in the dotenv the config points at.")
    if config.attach_key_file:
        if not config.attach_key_file.exists():
            # A server writes its local key on start and removes it on exit, so an absent
            # one usually means the instance is down rather than misconfigured. Say that,
            # or the next person goes hunting for a key problem that does not exist.
            raise BasicKBError(
                f"attach_cli.key_file {config.attach_key_file} does not exist. If that is a "
                f"server's local key, the served instance is probably not running — start it, "
                f"or use --no-attach to work on the store directly.")
        try:
            key = config.attach_key_file.read_text(encoding="utf-8").strip() or None
        except OSError as e:
            raise BasicKBError(
                f"attach_cli.key_file is {config.attach_key_file}, which cannot be read: {e}") from e
        if key:
            return key
        raise BasicKBError(f"attach_cli.key_file {config.attach_key_file} is empty.")
    return resolve_api_key(None) or LocalKey(config.store_dir).read()


def attach(
    config: Config,
    *,
    no_attach: bool = False,
    attach_url: Optional[str] = None,
    api_key: Optional[str] = None,
    on_note: Optional[Callable[[str], None]] = None,
) -> Optional[RemoteKnowledgeBase]:
    """A RemoteKnowledgeBase when this instance points at a served one, else None.

    Raises when a URL is in play and the server cannot be used: unreachable, unauthorised,
    or too far apart in version. `on_note` receives one line when the outcome is worth
    explaining, for stderr.
    """
    if no_attach or _env_says_no_attach():
        return None

    env_url = os.environ.get(ATTACH_URL_ENV, "").strip() or None
    url = attach_url or env_url or config.attach_url
    if not url:
        return None

    # Only the flag announces itself. The env var is what a machine is set to, so saying so
    # on every command would be noise on the one box that needs it most.
    if on_note is not None and attach_url and config.attach_url and attach_url != config.attach_url:
        on_note(f"--attach {url} overrides the configured {config.attach_url}")

    remote = RemoteKnowledgeBase(url, api_key=resolve_attach_key(config, api_key))
    remote.health(timeout=10.0)     # identity and version check; raises on anything wrong
    logger.info("attach: using the served instance at %s", url)
    return remote
