"""Renderers for the CLI: human text and JSON over the library's dataclasses.

The library returns data; these functions turn it into what a terminal reader or a
`--json` consumer sees. JSON and human text are two renderers over the same object,
neither derived from the other, so no type information is lost round-tripping through
a string. Nothing else in the package prints.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Optional

from .config import Config
from .freshness import FreshnessTracker
from .models import SearchResult
from .serialize import to_jsonable
from .sources import build_source


def emit_json(payload) -> None:
    """Print one JSON document to stdout.

    Callers must suppress progress output in this mode — a stray progress line on
    stdout makes the document unparseable.
    """
    print(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False))

def print_origin(kb) -> None:
    """One line naming where a command's answers came from: a served instance or this
    machine's own store."""
    url = getattr(kb, "url", None)
    if url:
        print(f"Reading : {url}  (attached)")
    else:
        print(f"Reading : {kb.store.path}  (local store)")


def ago(ts: Optional[float]) -> str:
    """Coarse age of a timestamp for a status row: minutes, hours or days.

    One unit, no calendar date — reading a status you want to know whether the index
    is an hour or a week behind, not which day the run happened.
    """
    if not ts:
        return "unknown"
    secs = max(0.0, time.time() - ts)
    if secs < 90:
        return "just now"
    if secs < 90 * 60:
        return f"{round(secs / 60)}m ago"
    if secs < 36 * 3600:
        return f"{round(secs / 3600)}h ago"
    return f"{round(secs / 86400)}d ago"

def print_results(hits: list[SearchResult], max_chars: int) -> None:
    """Human rendering of hits.

    Header: rank, title, `[source · content_type]`, scores. Second line: where the chunk
    came from — the url for web content, else the path inside the source — omitted when
    it would only repeat the title (LogSeq pages: title == filename stem).
    """
    reranked = any(r.rerank_score is not None for r in hits)
    print(f"Top {len(hits)} results{'  [reranked]' if reranked else ''}\n")

    for rank, r in enumerate(hits, 1):
        meta = r.metadata
        score_str = f"score={round(r.score, 3)}"
        if r.rerank_score is not None:
            score_str += f"  rerank={round(r.rerank_score, 4)}"

        title = meta.get("title", "?")
        tags = [meta.get("source", "?")]
        if meta.get("content_type", "unknown") != "unknown":
            tags.append(meta["content_type"])
        # Show the date only when there is one (transcripts have it; notes usually don't).
        date = meta.get("date")
        date_str = f"  ({date})" if date and date != "unknown" else ""
        header = f"[{rank}] {title}{date_str}  [{' · '.join(tags)}]  {score_str}"

        ref = meta.get("url") or meta.get("rel_path") or meta.get("file", "")
        stem = ref.rsplit("/", 1)[-1].rsplit(".", 1)[0] if ref else ""
        show_ref = bool(ref) and stem != title

        print("=" * 60)
        print(header)
        if show_ref:
            print(f"    {ref}")
        print("─" * 60)
        output = r.doc if not max_chars else r.doc[:max_chars]
        print(output)
        if max_chars and len(r.doc) > max_chars:
            print(f"  [...{len(r.doc) - max_chars} more chars]")
        print()

def print_status(st, nudges: Optional[int] = None) -> None:
    """Human rendering of one SourceStatus: a fixed set of rows, one line each.

    `nudges` is how many freshness reminders this source has produced since its last
    index; None when the reminder is switched off, which keeps it out of the output.
    """
    print(f"\n{'='*55}")
    print(f"Source  : {st.label}  ({st.source_id})")
    print(f"Store   : {st.store_dir}")
    print(f"Model   : {st.model_id}")

    if not st.directory_exists:
        print(f"  ⚠  SOURCE PATH MISSING: {st.directory}")
        print("     directory does not exist — moved, unmounted, or a broken symlink.")

    if not st.indexed or st.chunks == 0:
        print(f"Chunks  : 0")
        print(f"State   : {'not indexed' if not st.indexed else 'index empty'}")
        return

    print(f"Chunks  : {st.chunks:,}")
    print(f"Tokens  : ~{st.approx_tokens:,}  ({st.chars:,} chars)")
    print(f"Docs    : {st.docs_with_chunks:,} with chunks / {st.files_on_disk:,} files on disk")

    # One `State` row: the state, how much work is waiting, and how old the index is.
    # Files too short to chunk are tracked (hashed) but hold no chunks, so they never
    # count as missing — that is why Docs can be below files on disk.
    if st.files_on_disk == 0:
        why = "source path missing" if not st.directory_exists else "directory holds no matching files"
        state = f"⚠  0 files on disk ({why}); {st.docs_with_chunks:,} indexed docs orphaned"
    elif not st.tracked:
        state = "untracked — index once to enable change detection"
    elif st.stale:
        parts = [f"{n} {w}" for n, w in ((st.pending, "new"), (st.deleted, "deleted")) if n]
        if nudges:
            parts.append(f"nudges {nudges}")
        state = f"stale ({', '.join(parts)})"
    else:
        state = "up to date"
    print(f"State   : {state}. Last index: {ago(st.indexed_at)}")

    if st.date_min:
        print(f"Dates   : {st.date_min} → {st.date_max}")
    for ct, count in sorted(st.content_types.items()):
        print(f"  {ct}: {count:,} chunks")
    if st.oversized_chunks:
        print(f"  ⚠  oversized chunks: {st.oversized_chunks}")

def print_info(inf) -> None:
    """Human rendering of InstanceInfo. Compact on purpose — this is meant to be
    read at a glance before choosing a source to search."""
    print(f"\n{inf.name or '(unnamed instance)'}")
    print(f"  model : {inf.model_id}")
    print(f"  store : {inf.store_dir}")
    print(f"  totals: {len(inf.sources)} sources, {inf.total_files:,} files, {inf.total_chunks:,} chunks\n")

    for s in inf.sources:
        state = "" if s.indexed else "   [NOT INDEXED]"
        print(f"  {s.source_id}{state}")
        print(f"    {s.label}  ({s.type}, {s.chunker} chunker)")
        if s.description:
            print(f"    {s.description}")
        print(f"    {s.files:,} files -> {s.chunks:,} chunks  ({s.chunks_per_file} per file)\n")

def print_sources(config: Config) -> None:
    print(f"\nInstance '{config.name}' sources (use with --source):\n")
    for s in config.sources:
        try:
            src = build_source(s, config.base_dir)
            n_files = len(src.get_files())
            where = src.directory
        except Exception as e:  # bad source entry — report, don't hide
            print(f"  {s.get('id', '?'):<16}  [config error: {e}]")
            continue
        print(f"  {src.source_id:<16}  {src.label}  ({s.get('type', 'markdown')}, {n_files} files)")
        if src.description:
            print(f"                    {src.description}")
        print(f"                    {where}")
        print()
    print("  all               All sources combined (default)\n")

def print_inspect(config: Config) -> None:
    """Resolved runtime detail that reading the config file does not give you.

    One section for now: the freshness nudge exactly as it will be rendered, so you
    can see the built-in template when the config overrides nothing.
    """
    from .config import DEFAULT_FRESHNESS_MESSAGE

    print(f"\nInstance '{config.name or '(unnamed)'}'  ({config.path})")
    origin = "default" if config.freshness_message == DEFAULT_FRESHNESS_MESSAGE else "config override"
    print("\nFreshness nudge")
    print(f"  enabled           : {config.freshness_enabled}")
    print(f"  stale_after_days  : {config.freshness_stale_after_days:g}")
    print(f"  remind_every_days : {config.freshness_remind_every_days:g}")
    print(f"  template ({origin}):")
    print(f"    {config.freshness_message}")
    print("\n  placeholders: {source} {new} {updated} {deleted} {unchanged} {stale} "
          "{total} {days} {nudges}")
    print(f"  nudge state: {FreshnessTracker.from_config(config).path}")

def print_watch_event(ev) -> None:
    """Render one WatchEvent the way the foreground `watch` command always has."""
    err = dict(file=sys.stderr, flush=True)
    out = dict(flush=True)
    if ev.kind == "waiting":
        print(f"  ! {ev.message}", **err)
    elif ev.kind == "reconcile_skipped":
        print(f"  ! '{ev.source_id}': {ev.message}", **err)
    elif ev.kind == "reconcile":
        print(f"  reconcile {ev.source_id}: {ev.message}", **out)
    elif ev.kind == "missing_dir":
        print(f"  ! '{ev.source_id}': {ev.message}", **err)
    elif ev.kind == "watching":
        print(f"  watching {ev.source_id:<12} {ev.message}", **out)
    elif ev.kind == "nothing_to_watch":
        print(f"Nothing to watch ({ev.message}).", **err)
    elif ev.kind == "started":
        print(f"\nWatching {ev.paths} source(s). Edits reindex after their debounce. Ctrl-C to stop.\n", **out)
    elif ev.kind == "reindexed":
        print(f"  [{time.strftime('%H:%M:%S')}] reindexed {ev.source_id}: {ev.message} ({ev.paths} file(s))", **out)
    elif ev.kind == "error":
        print(f"  ! {ev.message} on '{ev.source_id}'", **err)
    elif ev.kind == "stopping":
        print("\nStopping — flushing pending reindexes...", **out)
    elif ev.kind == "stopped":
        print("Watch stopped.", **out)
