"""File-watching auto-reindex, as an object a process can start and stop.

One `Watcher` watches every enabled source directory recursively. File events feed a
single per-file debounce scheduler; when a file has been quiet for `debounce_seconds`
it is handed to one reindex worker, so the store only ever has one writer here.
Cross-platform via watchdog: FSEvents (macOS), ReadDirectoryChangesW (Windows), inotify
(Linux).

Lifecycle: `start()` reconciles offline edits and schedules the observers; `stop()`
flushes pending files and joins the worker; `run_forever()` does both around a sleep
loop for a foreground command. Everything the watcher does or notices is reported as a
`WatchEvent` to the `on_event` callback. Nothing here prints: the CLI renders events,
a server logs them.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .core import KnowledgeBase
from .errors import StoreError
from .models import ReindexResult
from .sources import DataSourceBase

logger = logging.getLogger("basic_kb")

# watchdog event types that mean "somebody read the file", not "the file changed".
_READ_ONLY_EVENTS = frozenset({"opened", "closed_no_write"})


def _relevant(path: str) -> bool:
    """Only markdown counts, and never editor scratch/temp files: a save's temp artifacts
    would otherwise cause churn. The real .md fires its own event."""
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
    """Resolved per-source watch options (engine defaults < per-source config < CLI flags)."""
    enabled: bool = True
    debounce_seconds: int = 30    # seconds of quiet before reindex; 0 = immediately


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


@dataclass
class WatchEvent:
    """One thing the watcher did or noticed.

    `kind` is one of: waiting (store holds another model; idling), reconcile,
    reconcile_skipped (guard tripped), missing_dir, watching, nothing_to_watch, started,
    reindexed, error, stopping, stopped. `message` is human-readable detail; `result`
    is set for reconcile and reindexed.
    """
    kind: str
    source_id: str = ""
    message: str = ""
    result: Optional[ReindexResult] = None
    paths: int = 0


OnEvent = Callable[[WatchEvent], None]


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
    on_event: Optional[OnEvent] = None
    _pending: dict[tuple[str, str], _Pending] = field(default_factory=dict)
    _cond: threading.Condition = field(default_factory=threading.Condition)
    _stop: bool = False

    def _emit(self, ev: WatchEvent) -> None:
        if self.on_event is not None:
            self.on_event(ev)

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
                res = self.kb.reindex_paths(
                    source, paths, self.chunk_size, self.overlap, self.min_chunk)
            except Exception as e:  # one bad batch must not kill the watcher
                logger.exception("watch reindex failed: source=%s", source.source_id)
                self._emit(WatchEvent("error", source.source_id, f"reindex error: {e}", paths=len(paths)))
                continue
            logger.info("watch reindexed source=%s %s (%d file(s))", source.source_id, res.summary(), len(paths))
            self._emit(WatchEvent("reindexed", source.source_id, res.summary(), result=res, paths=len(paths)))

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


class Watcher:
    """Watch a set of sources and keep their index current. See the module docstring."""

    def __init__(
        self,
        kb: KnowledgeBase,
        watched: list[tuple[DataSourceBase, WatchSettings]],
        chunk_size: int,
        overlap: int,
        min_chunk: int,
        *,
        guard: bool = True,
        guard_threshold: float = 0.9,
        on_event: Optional[OnEvent] = None,
        model_wait_s: float = 60,
    ) -> None:
        self.kb = kb
        self.watched = [(s, st) for s, st in watched if st.enabled]
        self.chunk = (chunk_size, overlap, min_chunk)
        self.guard = guard
        self.guard_threshold = guard_threshold
        self.on_event = on_event
        self.model_wait_s = model_wait_s
        self._engine = _Engine(kb, chunk_size, overlap, min_chunk, on_event=on_event)
        self._observer = None
        self._worker: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self.scheduled = 0

    def _emit(self, ev: WatchEvent) -> None:
        if self.on_event is not None:
            self.on_event(ev)

    def _wait_for_model(self) -> bool:
        """A watcher must never be the one to switch models. While the store holds another
        model's vectors (a rebuild in progress, or `index --switch-model` not run yet),
        idle and re-check instead of exiting, so a supervisor's Restart= does not crash-loop
        us. Returns False if stop() arrived while waiting."""
        while True:
            try:
                self.kb.prepare_model_switch([s for s, _ in self.watched], accept=False)
                return True
            except StoreError as e:
                logger.warning("watch waiting for store/model to match: %s", e)
                self._emit(WatchEvent("waiting", message=f"{e}; watcher idle, re-checking every {self.model_wait_s:g}s"))
                if self._stopping.wait(self.model_wait_s):
                    return False

    def _reconcile(self) -> None:
        """Embed anything that changed while the watcher was down."""
        chunk_size, overlap, min_chunk = self.chunk
        for source, _ in self.watched:
            stale = self.kb.stale_paths(source)
            if not stale:
                continue
            base = len(self.kb.store.manifest(source.source_id))
            # Reuse the corruption guard: a huge offline delta is likely a moved/broken
            # source, not real edits. Never re-embed it unattended.
            if self.guard and base and len(stale) / base >= self.guard_threshold and len(stale) >= 5:
                msg = (f"{len(stale)}/{base} files changed offline (>= {int(self.guard_threshold * 100)}%); "
                       f"skipping auto-reconcile. Run `basic-kb index --source {source.source_id} --yes` if this is real.")
                logger.warning("watch reconcile skipped by guard: source=%s stale=%d base=%d",
                               source.source_id, len(stale), base)
                self._emit(WatchEvent("reconcile_skipped", source.source_id, msg, paths=len(stale)))
                continue
            res = self.kb.reindex_paths(source, stale, chunk_size, overlap, min_chunk)
            self._emit(WatchEvent("reconcile", source.source_id, res.summary(), result=res, paths=len(stale)))

    def start(self) -> int:
        """Reconcile, then schedule the observers and start the worker. Returns how many
        source directories are being watched; 0 means there is nothing to do (every
        directory missing, or stop() arrived during the model wait)."""
        # Held until stop(): a CLI `index` in another shell now fails with StoreBusy instead
        # of racing this process. Reentrant, so a server that already holds it is fine.
        self.kb.writer_lock.acquire(timeout=0)
        if not self._wait_for_model():
            self.kb.writer_lock.release()
            return 0
        self._reconcile()

        from watchdog.observers import Observer
        self._observer = Observer()
        for source, settings in self.watched:
            if not source.directory.exists():
                self._emit(WatchEvent("missing_dir", source.source_id,
                                      f"{source.directory} does not exist; not watching it"))
                continue
            self._observer.schedule(_Handler(self._engine, source, settings.debounce_seconds),
                                    str(source.directory), recursive=True)
            self.scheduled += 1
            self._emit(WatchEvent("watching", source.source_id,
                                  f"{source.directory}  (debounce {settings.debounce_seconds}s)"))
        if not self.scheduled:
            self._emit(WatchEvent("nothing_to_watch", message="no enabled sources with an existing directory"))
            self._observer = None
            self.kb.writer_lock.release()
            return 0

        self._worker = threading.Thread(target=self._engine.run, name="reindex-worker", daemon=True)
        self._worker.start()
        self._observer.start()
        self._emit(WatchEvent("started", message=f"watching {self.scheduled} source(s)", paths=self.scheduled))
        return self.scheduled

    def stop(self, timeout: float = 120) -> None:
        """Stop observing, flush pending debounced files so nothing is lost, join the worker."""
        self._stopping.set()
        self._emit(WatchEvent("stopping", message="flushing pending reindexes"))
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
        self._engine.stop()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None
            self.kb.writer_lock.release()
        self._emit(WatchEvent("stopped"))

    def run_forever(self, poll_s: float = 1.0) -> None:
        """Foreground mode: start, sleep until KeyboardInterrupt, stop."""
        if not self.start():
            return
        try:
            while not self._stopping.wait(poll_s):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
