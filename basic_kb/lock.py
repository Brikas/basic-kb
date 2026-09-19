"""One writer per store, enforced by the operating system.

Every process that writes a store (`serve`, `watch`, `index`, `vacuum`, a module caller)
holds an advisory OS lock on `<store_dir>/writer.lock` while it writes; a server or
watcher holds it for its whole lifetime. The kernel releases the lock on every exit
path, including `kill -9`, an OOM kill and a power loss (after reboot nothing holds
it), so unlike a pid or state file it can never be stale. `filelock` supplies the
cross-platform primitive: `fcntl.flock` on POSIX, `msvcrt.locking` on Windows.

The lock object is reentrant within a process and shared across its threads
(`thread_local=False`), so a server that took the lock at startup can run `index()` from
a request-handler thread; in-process serialisation of writers is the KnowledgeBase's own
RLock, taken first. Advisory locks are unreliable on NFS and SMB mounts; a store has to
sit on a local disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from .errors import StoreError

LOCK_FILENAME = "writer.lock"


class StoreBusy(StoreError):
    """Another process holds the store's writer lock."""


class WriterLock:
    def __init__(self, store_dir: Path, timeout: float = 5.0) -> None:
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / LOCK_FILENAME
        self.timeout = timeout
        self._lock = FileLock(str(self.path), thread_local=False)

    @property
    def held(self) -> bool:
        """True while this object holds the lock (any depth)."""
        return self._lock.is_locked

    def acquire(self, timeout: Optional[float] = None) -> "WriterLock":
        """Take the lock, waiting up to `timeout` seconds (default: the constructor's).
        Raises StoreBusy when another process keeps it past that."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        wait = self.timeout if timeout is None else timeout
        try:
            self._lock.acquire(timeout=wait)
        except Timeout:
            raise StoreBusy(
                f"another process is writing the store in {self.store_dir} (it holds {self.path.name}). "
                f"If that is this instance's server, talk to it instead — give the config an "
                f"`attach_cli:` block, or pass --attach URL. Otherwise stop the other writer and retry."
            ) from None
        return self

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "WriterLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()

    def is_free(self) -> bool:
        """Probe without keeping the lock: take it for an instant through a separate
        handle and let go. False while any process, this one included, holds it."""
        if not self.path.parent.exists():
            return True
        probe = FileLock(str(self.path), thread_local=False)
        try:
            probe.acquire(timeout=0)
        except Timeout:
            return False
        probe.release()
        return True
