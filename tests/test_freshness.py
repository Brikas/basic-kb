"""FreshnessTracker: stale-age gate, remind window, counters, clearing, corrupt state."""
from __future__ import annotations

import json

from basic_kb.freshness import FreshnessSettings, FreshnessTracker
from basic_kb.models import ScanResult

DAY = 86400
CH = dict(chunk_size=400, overlap=40, min_chunk=20)


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(store_dir, clock, **kw) -> FreshnessTracker:
    return FreshnessTracker(store_dir, FreshnessSettings(**kw), now=clock)


def stale(kb, notes):
    (notes.directory / "coffee.md").write_text("# C\n\nchanged body with enough words in it.\n", encoding="utf-8")


def test_disabled_tracker_is_silent_and_writes_nothing(indexed_kb, notes, config):
    t = make(config.store_dir, Clock(), enabled=False)
    stale(indexed_kb, notes)
    assert t.evaluate(indexed_kb, [notes]) == []
    assert not t.path.exists()


def test_clean_source_leaves_no_state(indexed_kb, notes, config):
    t = make(config.store_dir, Clock())
    assert t.evaluate(indexed_kb, [notes]) == []
    assert t.state() == {} and t.nudges_for("notes") == 0


def test_nudge_only_after_stale_age_then_once_per_window(indexed_kb, notes, config):
    clock = Clock()
    t = make(config.store_dir, clock, stale_after_days=3, remind_every_days=1)
    stale(indexed_kb, notes)

    assert t.evaluate(indexed_kb, [notes]) == []            # first sighting: clock starts
    assert t.state()["notes"]["nudges"] == 0
    clock.t += 2 * DAY
    assert t.evaluate(indexed_kb, [notes]) == []            # 2 days: still under the threshold
    clock.t += 1.5 * DAY
    msgs = t.evaluate(indexed_kb, [notes])                  # 3.5 days: first nudge
    assert len(msgs) == 1 and "reminder 1" in msgs[0] and "'notes'" in msgs[0]
    assert t.nudges_for("notes") == 1
    clock.t += 0.5 * DAY
    assert t.evaluate(indexed_kb, [notes]) == []            # inside the remind window: quiet
    clock.t += DAY
    assert "reminder 2" in t.evaluate(indexed_kb, [notes])[0]


def test_reindexing_resets_the_clock(indexed_kb, notes, config):
    clock = Clock()
    t = make(config.store_dir, clock, stale_after_days=0, remind_every_days=0)
    stale(indexed_kb, notes)
    assert len(t.evaluate(indexed_kb, [notes])) == 1
    indexed_kb.index(notes, **CH)
    assert t.evaluate(indexed_kb, [notes]) == []
    assert "notes" not in t.state()


def test_clear_removes_only_named_sources(config):
    t = make(config.store_dir, Clock())
    t._write({"a": {"nudges": 2}, "b": {"nudges": 1}})
    assert t.clear(["a", "zzz"]) == ["a"]
    assert t.state() == {"b": {"nudges": 1}}
    assert t.clear(["b"]) == ["b"]
    t.path.unlink()
    assert t.clear(["b"]) == []


def test_corrupt_state_reads_as_empty(config):
    t = make(config.store_dir, Clock())
    t.path.parent.mkdir(parents=True, exist_ok=True)
    t.path.write_text("{not json", encoding="utf-8")
    assert t.state() == {}
    t.path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert t.state() == {}


def test_legacy_state_without_counter_reads_as_zero(config):
    t = make(config.store_dir, Clock())
    t._write({"notes": 123.0})
    assert t.nudges_for("notes") == 0


def test_message_formatting_and_bad_placeholder():
    res = ScanResult("notes", "N", True, 10, 2, 1, 6, 1)   # new, updated, unchanged, deleted
    good = FreshnessSettings(message="{source}: {stale}/{total} stale, day {days}, nudge {nudges}")
    assert good.format(res, 3) == "notes: 4/10 stale, day 3, nudge 3"
    bad = FreshnessSettings(message="{source} {nope}")
    out = bad.format(res, 1)
    assert "invalid placeholder" in out and "'notes' is stale (4 file(s))" in out


def test_from_config(config):
    t = FreshnessTracker.from_config(config)
    assert t.settings.enabled is False and t.path == config.store_dir / "freshness_state.json"
