"""Freshness nudges: remind a caller about sources that have stayed stale for a while.

A source must be *continuously* stale for `stale_after_days` before the first nudge;
after that it re-nags at most once per `remind_every_days` for as long as it stays
un-indexed. Re-indexing (the source goes clean) clears its entry, so it must age past
the threshold again before it can nag.

Per-source state is `{first_stale, last_eval, nudges}` in `<store_dir>/freshness_state.json`.
The `last_eval` gate means a source is re-scanned at most once per remind window, which
both bounds the hashing cost and gives the nag cadence. `nudges` counts reminders emitted
since the last index, so a rising number shows a nudge being ignored.

The tracker decides and returns messages; the caller renders them. The CLI prints them
to stderr, a server returns them in the response, and `index` clears the entries for the
sources it rebuilt. Nothing here prints.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import DEFAULT_FRESHNESS_MESSAGE, Config
from .models import ScanResult

logger = logging.getLogger("basic_kb")

STATE_FILENAME = "freshness_state.json"


@dataclass
class FreshnessSettings:
    """The `freshness:` config block."""
    enabled: bool = True
    stale_after_days: float = 3
    remind_every_days: float = 1
    message: str = DEFAULT_FRESHNESS_MESSAGE

    @classmethod
    def from_config(cls, config: Config) -> "FreshnessSettings":
        return cls(enabled=config.freshness_enabled,
                   stale_after_days=config.freshness_stale_after_days,
                   remind_every_days=config.freshness_remind_every_days,
                   message=config.freshness_message)

    def format(self, res: ScanResult, nudges: int) -> str:
        """Render the nudge for one stale source. An invalid placeholder in a custom
        template produces a message that says so and still names the stale source."""
        fields = dict(source=res.source_id, new=res.new, updated=res.updated,
                      deleted=res.deleted, unchanged=res.unchanged, stale=res.stale,
                      total=res.files_on_disk, days=int(self.stale_after_days), nudges=nudges)
        try:
            return self.message.format(**fields)
        except (KeyError, IndexError) as e:
            return (f"[basic-kb] freshness message has an invalid placeholder {e}; "
                    f"valid: {', '.join(fields)}. Source '{res.source_id}' is stale "
                    f"({res.stale} file(s)).")


class FreshnessTracker:
    """Owns the state file for one instance. `now` is injectable for tests."""

    def __init__(self, store_dir: Path, settings: FreshnessSettings,
                 now: Callable[[], float] = time.time) -> None:
        self.settings = settings
        self.path = Path(store_dir) / STATE_FILENAME
        self._now = now

    @classmethod
    def from_config(cls, config: Config) -> "FreshnessTracker":
        return cls(config.store_dir, FreshnessSettings.from_config(config))

    def state(self) -> dict:
        """The per-source state map; `{}` when there is no file. A corrupt file reads as
        empty and is logged: nudge counts start over, search is never blocked by it."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("freshness state at %s is not valid JSON; nudge counts start over", self.path)
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state), encoding="utf-8")

    @staticmethod
    def _nudges(state: dict, source_id: str) -> int:
        entry = state.get(source_id)
        return int(entry.get("nudges", 0)) if isinstance(entry, dict) else 0

    def nudges_for(self, source_id: str) -> int:
        """Reminders emitted for this source since its last index."""
        return self._nudges(self.state(), source_id)

    def clear(self, source_ids: Iterable[str]) -> list[str]:
        """Forget the given sources (an index run is what a nudge asks for). Returns the ids
        that had an entry."""
        if not self.path.exists():
            return []
        state = self.state()
        removed = [sid for sid in source_ids if state.pop(sid, None) is not None]
        if removed:
            self._write(state)
        return removed

    def evaluate(self, kb, sources) -> list[str]:
        """Scan the sources that are due, update the state, return the nudges to show."""
        if not self.settings.enabled:
            return []
        now = self._now()
        stale_after = max(0.0, self.settings.stale_after_days) * 86400
        remind_every = max(0.0, self.settings.remind_every_days) * 86400
        state = self.state()

        messages: list[str] = []
        dirty = False
        for s in sources:
            entry = state.get(s.source_id)
            entry = entry if isinstance(entry, dict) else {}
            if now - float(entry.get("last_eval", 0)) < remind_every:
                continue   # evaluated within this window: no re-scan, no re-nag yet
            dirty = True
            res = kb.scan(s)
            if not (res.tracked and res.stale):
                state.pop(s.source_id, None)   # clean or freshly re-indexed: reset the clock
                continue
            first_stale = float(entry.get("first_stale") or now)   # start counting on first sighting
            nudges = self._nudges(state, s.source_id)
            if now - first_stale >= stale_after:
                nudges += 1
                messages.append(self.settings.format(res, nudges))
            state[s.source_id] = {"first_stale": first_stale, "last_eval": now, "nudges": nudges}

        if dirty:
            self._write(state)
        return messages
