"""File-watching auto-reindex — the foreground `basic-kb watch` command.

One run watches every enabled source directory recursively. File events feed a single
per-file debounce scheduler; when a file has been quiet for `debounce_seconds` it is
handed to a single reindex worker, so the store only ever has one writer. Cross-platform
via watchdog: FSEvents (macOS), ReadDirectoryChangesW (Windows), inotify (Linux), plus
a polling backend for filesystems that don't emit native events.

Lifecycle: startup reconcile (catch edits made while the watcher was off) → watch until
Ctrl-C → on shutdown, flush any pending debounced files so nothing is lost.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .core import KnowledgeBase
from .errors import StoreError
from .sources import DataSourceBase

logger = logging.getLogger("basic_kb")

# watchdog event types that mean "somebody read the file", not "the file changed".
_READ_ONLY_EVENTS = frozenset({"opened", "closed_no_write"})

# Ignore editor scratch/temp files so a save's temp artifacts don't cause churn.
# The real file (its .md) fires its own event and is what we reindex.
def _relevant(path: str) -> bool:
    name = Path(path).name
    if not name.lower().endswith(".md"):
        return False
    if name.startswith(".") or name.startswith("~") or name.endswith("~"):
        return False
    if name.endswith(".tmp") or name.endswith(".swp") or name.endswith(".swx"):
        return False
    return True


@dataclass
class WatchSettings:
    """Resolved per-source watch options (config defaults < per-source < CLI flags)."""
    enabled: bool = True
    debounce_seconds: int = 30    # seconds of quiet before reindex; 0 = immediately

# Note: a polling backend (watchdog's PollingObserver, which diffs the filesystem on a
# timer) was considered for filesystems that don't emit native events. Dropped as
# unnecessary — native events proved reliable on the target filesystems, including
# Google Drive. Re-add via a per-source `mode` if a source ever needs it.


@dataclass
class _Pending:
    due: float                    # monotonic time this file becomes reindex-ready
    source: DataSourceBase
    path: Path


@dataclass
class _Engine:
    """Owns the pending map and the single reindex worker thread.

    Observer threads only ever call notify() (cheap, under lock); all store writes
    happen on this one worker thread, so there is never concurrent access.
    """
    kb: KnowledgeBase
    chunk_size: int
    overlap: int
    min_chunk: int
    _pending: dict[tuple[str, str], _Pending] = field(default_factory=dict)
    _cond: threading.Condition = field(default_factory=threading.Condition)
    _stop: bool = False

    def notify(self, source: DataSourceBase, path: Path, debounce: int) -> None:
        key = (source.source_id, str(path))
        with self._cond:
            self._pending[key] = _Pending(time.monotonic() + max(0, debounce), source, path)
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                if self._stop and not self._pending:
                    return
                now = time.monotonic()
                # On shutdown, flush everything now regardless of debounce.
                due = [(k, p) for k, p in self._pending.items() if self._stop or p.due <= now]
                if not due:
                    nxt = min((p.due for p in self._pending.values()), default=None)
                    timeout = None if nxt is None else max(0.0, nxt - now)
                    self._cond.wait(timeout)
                    continue
                for k, _ in due:
                    del self._pending[k]
            self._reindex(due)

    def _reindex(self, due: list[tuple[tuple[str, str], _Pending]]) -> None:
        by_source: dict[str, tuple[DataSourceBase, list[Path]]] = {}
        for _, p in due:
            entry = by_source.setdefault(p.source.source_id, (p.source, []))
            entry[1].append(p.path)
        for source, paths in by_source.values():
            try:
                counts = self.kb.reindex_paths(
                    source, paths, self.chunk_size, self.overlap, self.min_chunk)
            except Exception as e:  # one bad batch must not kill the watcher
                logger.exception("watch reindex failed: source=%s", source.source_id)
                print(f"  ! reindex error on '{source.source_id}': {e}", file=sys.stderr, flush=True)
                continue
            done = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
            print(f"  [{time.strftime('%H:%M:%S')}] reindexed {source.source_id}: "
                  f"{done or 'no change'} ({len(paths)} file(s))", flush=True)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()


class _Handler:
    """watchdog event handler bound to one source; forwards relevant files to the engine."""

    def __init__(self, engine: _Engine, source: DataSourceBase, debounce: int) -> None:
        self.engine, self.source, self.debounce = engine, source, debounce

    def dispatch(self, event) -> None:
        if getattr(event, "is_directory", False):
            return
        # Only content changes count. Linux inotify also reports plain reads (opened /
        # closed_no_write); reacting to those makes the watcher re-trigger itself — its own
        # reindex reads the file — and loop forever at the debounce period.
        if getattr(event, "event_type", None) in _READ_ONLY_EVENTS:
            return
        # Moves report both src (gone) and dest (the new file); handle whichever is a note.
        for attr in ("src_path", "dest_path"):
            p = getattr(event, attr, None)
            if p and _relevant(p) and not self.source.is_excluded(Path(p)):
                self.engine.notify(self.source, Path(p), self.debounce)


def resolve_settings(raw_source: dict, debounce_override: Optional[int]) -> WatchSettings:
    """Watch options are PER-SOURCE: read this source's `watch:` block, falling back to
    the engine defaults (WatchSettings). A CLI --debounce overrides all sources for the run."""
    sw = raw_source.get("watch", {}) or {}
    default = WatchSettings()
    enabled = bool(sw.get("enabled", default.enabled))
    debounce = int(sw.get("debounce_seconds", default.debounce_seconds))
    if debounce_override is not None:
        debounce = debounce_override
    return WatchSettings(enabled=enabled, debounce_seconds=debounce)


def run_watch(kb: KnowledgeBase, watched: list[tuple[DataSourceBase, WatchSettings]],
              config, chunk_size: int, overlap: int, min_chunk: int) -> None:
    """Reconcile offline edits, then watch until interrupted. `watched` is already
    filtered to enabled sources."""
    engine = _Engine(kb, chunk_size, overlap, min_chunk)

    # A watcher must never be the one to switch models. While a model switch or rebuild
    # is in progress (or nobody has run `index --force --source all` yet), the store holds
    # another model's vectors. Exiting here would make a supervisor (systemd Restart=)
    # crash-loop us; instead wait and re-check, so the watcher picks up on its own once
    # the rebuild lands. The message says exactly what to run.
    wait_s = 60
    while True:
        try:
            kb.prepare_model_switch([s for s, _ in watched], accept=False)
            break
        except StoreError as e:
            print(f"  ! {e}\n  ! watcher idle; re-checking every {wait_s}s.", file=sys.stderr, flush=True)
            logger.warning("watch waiting for store/model to match: %s", e)
            time.sleep(wait_s)

    # 1) Startup reconcile: embed anything that changed while the watcher was down.
    for source, _ in watched:
        stale = kb.stale_paths(source)
        if not stale:
            continue
        base = len(kb.store.manifest(source.source_id))
        # Reuse the corruption guard: a huge offline delta is likely a moved/broken
        # source, not real edits — don't silently re-embed it unattended.
        if (config.reindex_guard and base and
                len(stale) / base >= config.reindex_guard_threshold and
                len(stale) >= 5):
            print(f"  ! '{source.source_id}': {len(stale)}/{base} files changed offline "
                  f"(>= {int(config.reindex_guard_threshold*100)}%). Skipping auto-reconcile — "
                  f"run `basic-kb index --source {source.source_id} --yes` if this is real.",
                  file=sys.stderr, flush=True)
            logger.warning("watch reconcile skipped by guard: source=%s stale=%d base=%d",
                           source.source_id, len(stale), base)
            continue
        counts = kb.reindex_paths(source, stale, chunk_size, overlap, min_chunk)
        print(f"  reconcile {source.source_id}: "
              f"{', '.join(f'{k}={v}' for k, v in counts.items() if v) or 'no change'}", flush=True)

    # 2) Schedule watches on one native observer (OS file events).
    from watchdog.observers import Observer
    observer = Observer()
    scheduled = 0
    for source, settings in watched:
        if not source.directory.exists():
            print(f"  ! '{source.source_id}': {source.directory} does not exist — not watching it.",
                  file=sys.stderr, flush=True)
            continue
        observer.schedule(_Handler(engine, source, settings.debounce_seconds),
                          str(source.directory), recursive=True)
        scheduled += 1
        print(f"  watching {source.source_id:<12} {source.directory}  "
              f"(debounce {settings.debounce_seconds}s)", flush=True)

    if not scheduled:
        print("Nothing to watch (no enabled sources with an existing directory).", file=sys.stderr)
        return

    worker = threading.Thread(target=engine.run, name="reindex-worker", daemon=True)
    worker.start()
    observer.start()

    print(f"\nWatching {scheduled} source(s). Edits reindex after their debounce. "
          f"Ctrl-C to stop.\n", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping — flushing pending reindexes...", flush=True)
    finally:
        observer.stop()
        observer.join()
        engine.stop()
        worker.join(timeout=120)
        print("Watch stopped.", flush=True)
