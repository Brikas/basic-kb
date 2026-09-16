"""API keys for the served API (ADR 0002): flat bearer keys, several at once, local admin.

`<store_dir>/api-keys.json` holds one record per key: id, name, prefix (for display),
sha256 of the key, created_at, revoked_at. The plaintext is shown once at creation and
never stored. Verification hashes the presented key and compares it to every active
record with a constant-time comparison. The file is re-read when its mtime changes, so
a revocation takes effect in a running server without a restart.

Local attach (ADR 0002): a server with authentication on mints one ephemeral key per
start and writes it to `<store_dir>/local.key` with owner-only permissions. A CLI on the
same machine reads it; the trust boundary is the directory's permissions, the same
boundary that already guards the database.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .errors import BasicKBError

logger = logging.getLogger("basic_kb")

KEYS_FILENAME = "api-keys.json"
LOCAL_KEY_FILENAME = "local.key"
KEY_PREFIX = "bkb_"


class KeyError_(BasicKBError):
    """Key management failed: unknown id, duplicate name, unreadable file."""


@dataclass
class ApiKey:
    """One stored key. `sha256` is the only secret-derived field and is one-way."""
    id: str
    name: str
    prefix: str            # first characters of the plaintext, for identification
    sha256: str
    created_at: float
    revoked_at: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _write_private(path: Path, text: str) -> None:
    """Write a file readable by the owner only, atomically (temp file, then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError as e:          # Windows has no POSIX mode bits; the ACL is the owner's
        logger.debug("chmod on %s skipped: %s", path, e)


def generate_key() -> str:
    """`bkb_` plus 32 URL-safe characters (192 bits of randomness)."""
    return KEY_PREFIX + secrets.token_urlsafe(24)


class ApiKeyStore:
    """The key file for one instance."""

    def __init__(self, store_dir: Path) -> None:
        self.path = Path(store_dir) / KEYS_FILENAME
        self._cache: list[ApiKey] = []
        self._stamp: Optional[tuple] = None    # (inode, mtime_ns, size) of the file the cache came from

    # --- reading -------------------------------------------------------------------------

    def _load(self) -> list[ApiKey]:
        if not self.path.exists():
            self._cache, self._stamp = [], None
            return self._cache
        # Every write replaces the file (new inode), and Linux mtime granularity is coarse
        # enough for two writes to share a timestamp; the inode makes the stamp reliable.
        st = self.path.stat()
        stamp = (st.st_ino, st.st_mtime_ns, st.st_size)
        if self._stamp == stamp:
            return self._cache
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise KeyError_(f"API key file {self.path} is not valid JSON: {e}") from e
        self._cache = [ApiKey(**rec) for rec in data.get("keys", [])]
        self._stamp = stamp
        return self._cache

    def list(self) -> list[ApiKey]:
        """Every key, active and revoked, oldest first."""
        return list(self._load())

    def active(self) -> list[ApiKey]:
        return [k for k in self._load() if k.active]

    def verify(self, presented: str) -> Optional[ApiKey]:
        """The active key matching `presented`, or None. Constant-time per candidate."""
        if not presented:
            return None
        digest = _hash(presented)
        for k in self._load():
            if k.active and hmac.compare_digest(k.sha256, digest):
                return k
        return None

    # --- writing -------------------------------------------------------------------------

    def _save(self, keys: list[ApiKey]) -> None:
        _write_private(self.path, json.dumps({"keys": [asdict(k) for k in keys]}, indent=2))
        self._stamp = None            # force a re-read on the next call

    def create(self, name: str) -> tuple[ApiKey, str]:
        """Mint a key. Returns (record, plaintext); the plaintext is never stored."""
        name = name.strip()
        if not name:
            raise KeyError_("a key needs a name (who or what will use it)")
        keys = self._load()
        if any(k.name == name and k.active for k in keys):
            raise KeyError_(f"an active key named {name!r} already exists; revoke it or pick another name")
        plaintext = generate_key()
        record = ApiKey(id=secrets.token_hex(4), name=name, prefix=plaintext[:len(KEY_PREFIX) + 6],
                        sha256=_hash(plaintext), created_at=time.time())
        self._save(keys + [record])
        logger.info("api key created: id=%s name=%s", record.id, record.name)
        return record, plaintext

    def revoke(self, ident: str) -> ApiKey:
        """Revoke by id, or by name when that names exactly one active key."""
        keys = self._load()
        matches = [k for k in keys if k.active and (k.id == ident or k.name == ident)]
        if not matches:
            raise KeyError_(f"no active key with id or name {ident!r}")
        if len(matches) > 1:
            raise KeyError_(f"{ident!r} names {len(matches)} active keys; revoke by id: "
                            + ", ".join(k.id for k in matches))
        matches[0].revoked_at = time.time()
        self._save(keys)
        logger.info("api key revoked: id=%s name=%s", matches[0].id, matches[0].name)
        return matches[0]


class LocalKey:
    """The per-start key a server leaves for same-machine CLI attach."""

    def __init__(self, store_dir: Path) -> None:
        self.path = Path(store_dir) / LOCAL_KEY_FILENAME

    def write(self) -> str:
        key = generate_key()
        _write_private(self.path, key)
        return key

    def read(self) -> Optional[str]:
        if not self.path.exists():
            return None
        return self.path.read_text(encoding="utf-8").strip() or None

    def remove(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
