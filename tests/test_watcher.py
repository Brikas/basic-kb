"""Watcher: event filtering, per-source settings, the debounce engine, and the real thing
end to end on a temp directory with watchdog's native observer."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from basic_kb.core import KnowledgeBase
from basic_kb.models import ReindexResult
from basic_kb.sources import MarkdownSource
from basic_kb.watcher import WatchEvent, Watcher, WatchSettings, _Engine, _Handler, _relevant, resolve_settings

from .conftest import write_tree
from .fakes import FakeEmbedder

CH = dict(chunk_size=400, overlap=40, min_chunk=20)


def test_relevant_filters_temp_and_non_markdown():
    assert _relevant("/x/note.md") is True
    assert _relevant("/x/Note.MD") is True
    assert _relevant("/x/note.txt") is False
    assert _relevant("/x/.hidden.md") is False
    assert _relevant("/x/~lock.md") is False
    assert _relevant("/x/note.md~") is False
    assert _relevant("/x/note.md.swp") is False


def test_resolve_settings_defaults_per_source_and_override():
    assert resolve_settings({}, None) == WatchSettings(enabled=True, debounce_seconds=30)
    assert resolve_settings({"watch": {"enabled": False, "debounce_seconds": 5}}, None) == WatchSettings(False, 5)
    assert resolve_settings({"watch": {"debounce_seconds": 5}}, 0).debounce_seconds == 0


# --- handler -------------------------------------------------------------------------------------------

class RecordingEngine:
    def __init__(self):
        self.notified: list[tuple[str, str, int]] = []

    def notify(self, source, path, debounce):
        self.notified.append((source.source_id, Path(path).name, debounce))


def ev(event_type: str, src_path: str, dest_path: str | None = None, is_directory: bool = False):
    return SimpleNamespace(event_type=event_type, src_path=src_path, dest_path=dest_path, is_directory=is_directory)


def test_handler_ignores_reads_and_directories(tmp_path: Path):
    """Regression for the 2026-08-29 self-triggering loop: inotify read events must not requeue."""
    src = MarkdownSource("n", tmp_path, exclude=["drafts/"])
    eng = RecordingEngine()
    h = _Handler(eng, src, debounce=3)
    h.dispatch(ev("opened", str(tmp_path / "a.md")))
    h.dispatch(ev("closed_no_write", str(tmp_path / "a.md")))
    h.dispatch(ev("modified", str(tmp_path / "sub"), is_directory=True))
    assert eng.notified == []


def test_handler_forwards_content_events_and_moves(tmp_path: Path):
    src = MarkdownSource("n", tmp_path, exclude=["drafts/"])
    eng = RecordingEngine()
    h = _Handler(eng, src, debounce=3)
    h.dispatch(ev("modified", str(tmp_path / "a.md")))
    h.dispatch(ev("moved", str(tmp_path / "old.md"), str(tmp_path / "new.md")))
    h.dispatch(ev("modified", str(tmp_path / "a.txt")))              # not markdown
    h.dispatch(ev("modified", str(tmp_path / "drafts" / "d.md")))    # excluded
    assert eng.notified == [("n", "a.md", 3), ("n", "old.md", 3), ("n", "new.md", 3)]


# --- engine ----------------------------------------------------------------------------------------------

class RecordingKB:
    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def reindex_paths(self, source, paths, chunk_size, overlap, min_chunk):
        self.calls.append((source.source_id, sorted(Path(p).name for p in paths)))
        return ReindexResult(source.source_id, embedded=len(paths), chunks_embedded=len(paths))


def run_engine(eng: _Engine) -> None:
    t = threading.Thread(target=eng.run, daemon=True)
    t.start()
    eng.stop()
    t.join(timeout=5)
    assert not t.is_alive()


def test_engine_groups_due_files_per_source_and_flushes_on_stop(tmp_path: Path):
    kb, events = RecordingKB(), []
    eng = _Engine(kb, 400, 40, 20, on_event=events.append)
    a = MarkdownSource("a", tmp_path / "a")
    b = MarkdownSource("b", tmp_path / "b")
    eng.notify(a, tmp_path / "a" / "one.md", debounce=0)
    eng.notify(a, tmp_path / "a" / "two.md", debounce=0)
    eng.notify(a, tmp_path / "a" / "one.md", debounce=0)      # same key: coalesced
    eng.notify(b, tmp_path / "b" / "x.md", debounce=3600)     # far in the future: flushed by stop()
    run_engine(eng)
    assert sorted(kb.calls) == [("a", ["one.md", "two.md"]), ("b", ["x.md"])]
    kinds = sorted((e.kind, e.source_id, e.paths) for e in events)
    assert kinds == [("reindexed", "a", 2), ("reindexed", "b", 1)]
    assert events[0].result.embedded == events[0].paths


def test_engine_survives_a_failing_reindex(tmp_path: Path):
    class Boom:
        def reindex_paths(self, *a, **k):
            raise RuntimeError("kaput")
    events = []
    eng = _Engine(Boom(), 400, 40, 20, on_event=events.append)
    eng.notify(MarkdownSource("a", tmp_path), tmp_path / "x.md", 0)
    run_engine(eng)
    assert [e.kind for e in events] == ["error"] and "kaput" in events[0].message


# --- Watcher end to end ------------------------------------------------------------------------------------

def wait_for(events: list, kind: str, timeout: float = 10.0) -> WatchEvent:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for e in events:
            if e.kind == kind:
                return e
        time.sleep(0.05)
    raise AssertionError(f"no {kind!r} event within {timeout}s; got {[e.kind for e in events]}")


@pytest.fixture
def watcher_kb(indexed_kb, notes):
    return indexed_kb


def test_watcher_reconciles_then_reacts_to_edits(watcher_kb, notes):
    events: list[WatchEvent] = []
    coffee = notes.directory / "coffee.md"
    coffee.write_text("# Coffee\n\nEdited offline while the watcher was down, long enough to chunk.\n", encoding="utf-8")
    w = Watcher(watcher_kb, [(notes, WatchSettings(True, 0))], **CH, on_event=events.append)
    try:
        assert w.start() == 1
        rec = wait_for(events, "reconcile")
        assert rec.source_id == "notes" and rec.result.embedded == 1
        assert [e.kind for e in events][-2:] == ["watching", "started"]

        (notes.directory / "fresh.md").write_text("# Fresh\n\nA brand new note typed while watching, long enough.\n",
                                                 encoding="utf-8")
        done = wait_for(events, "reindexed")
        assert done.source_id == "notes" and done.result.embedded >= 1
        assert "fresh.md" in watcher_kb.store.manifest("notes")
    finally:
        w.stop()
    assert [e.kind for e in events][-2:] == ["stopping", "stopped"]


def test_watcher_skips_disabled_and_missing_sources(watcher_kb, notes, tmp_path):
    events: list[WatchEvent] = []
    gone = MarkdownSource("gone", tmp_path / "nowhere")
    w = Watcher(watcher_kb, [(notes, WatchSettings(False, 0)), (gone, WatchSettings(True, 0))], **CH,
                on_event=events.append)
    assert w.start() == 0
    assert [e.kind for e in events] == ["missing_dir", "nothing_to_watch"]
    w.stop()


def test_watcher_reconcile_respects_the_guard(kb, tmp_path):
    write_tree(tmp_path / "many", {f"n{i}.md": f"# N{i}\n\nEnough words here to make a chunk for note {i}.\n" for i in range(6)})
    src = MarkdownSource("many", tmp_path / "many", chunker="recursive")
    kb.index(src, **CH)
    for f in src.get_files():
        f.write_text(f.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    events: list[WatchEvent] = []
    w = Watcher(kb, [(src, WatchSettings(True, 0))], **CH, on_event=events.append)
    try:
        w.start()
        skipped = wait_for(events, "reconcile_skipped")
        assert skipped.paths == 6 and "--yes" in skipped.message
        assert kb.scan(src).updated == 6                    # nothing was embedded
    finally:
        w.stop()


def test_watcher_waits_when_store_holds_another_model_and_stops_cleanly(indexed_kb, notes, config):
    other = KnowledgeBase(FakeEmbedder(dim=16, model_id="other"), config.store_dir)
    events: list[WatchEvent] = []
    w = Watcher(other, [(notes, WatchSettings(True, 0))], **CH, on_event=events.append, model_wait_s=0.2)
    result: list[int] = []
    t = threading.Thread(target=lambda: result.append(w.start()), daemon=True)
    t.start()
    wait_for(events, "waiting", timeout=5)
    w.stop()
    t.join(timeout=5)
    assert result == [0]
    assert "Model switch detected" in events[0].message
