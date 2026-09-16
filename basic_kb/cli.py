"""basic-kb command-line interface.

  python -m basic_kb index  --config CFG [--source ID|all] [--force] [--preview]
  python -m basic_kb search "..." --config CFG [--source ID|all] [--n N]
  python -m basic_kb status --config CFG [--source ID|all]
  python -m basic_kb watch  --config CFG [--source ID|all] [--debounce SEC]

Config is resolved automatically: --config flag > $BASIC_KB_CONFIG > a basic-kb.yaml
found by walking up from the current directory (see README).
Multi-query search, merged into one ranked list (re-framings of one need = better recall):
  python -m basic_kb search "price too high" "budget concern" --config CFG
Batch mode — each query returns its own top-n block (different needs in one call):
  python -m basic_kb search "coffee gear" "tax deadlines" --separate --config CFG

Embedding / chunking overrides (override config defaults for one run):
  --model NAME  --chunk-size N  --overlap N  --min-chunk N

Reranking (search only; on when the config's `reranker:` block selects a backend):
  --no-rerank | --rerank (strict)  --reranker BACKEND  --reranker-model M  --rerank-candidates N

Secrets: a cloud reranker reads the API key named by its `api_key_env:` (see
rerankers.py). Set it in the environment, pass --env-file PATH, or set `env_file:`
in the config (CLI/shell win over the config file).
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Optional

from .attach import attach
from .config import Config, find_config, load_config, load_env_file
from .errors import BasicKBError, UnknownSource
from .freshness import FreshnessTracker
from .keys import ApiKeyStore
from .version import __version__
from .core import DEFAULT_N, KnowledgeBase, cores_to_threads, lower_process_priority, setup_file_logging
from .embedders import FastEmbedEmbedder
from .render import (
    emit_json, print_info, print_inspect, print_results, print_sources, print_status, print_watch_event,
)
from .rerankers import RERANKER_TYPES
from .sources import DataSourceBase, resolve_sources


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _effective(args: argparse.Namespace, config: Config) -> tuple[str, int, int, int]:
    """Resolve model + chunk params: CLI flag wins, else config default."""
    model = getattr(args, "model", None) or config.embedding_model
    chunk_size = getattr(args, "chunk_size", None) or config.chunk_size
    overlap = getattr(args, "overlap", None) if getattr(args, "overlap", None) is not None else config.overlap
    min_chunk = getattr(args, "min_chunk", None) or config.min_chunk
    return model, chunk_size, overlap, min_chunk


def _attached(args: argparse.Namespace, config: Config):
    """The served instance for this config when one is alive and attach is allowed, else None.
    Anything worth knowing about the decision goes to stderr as one `[basic-kb]` line."""
    return attach(
        config,
        no_attach=getattr(args, "no_attach", False),
        attach_url=getattr(args, "attach", None),
        api_key=getattr(args, "api_key", None),
        model_override=getattr(args, "model", None),
        on_note=lambda m: print(f"[basic-kb] {m}", file=sys.stderr),
    )


def _kb_for(args: argparse.Namespace, config: Config, *, search: bool, threads: Optional[int] = None):
    """Attached RemoteKnowledgeBase when possible, else a local KnowledgeBase (with a reranker
    only when `search`, the one operation that uses it)."""
    remote = _attached(args, config)
    if remote is not None:
        return remote
    return _build_kb(args, config, threads=threads) if search else _scan_kb(args, config, threads=threads)


def _build_kb(args: argparse.Namespace, config: Config, threads: Optional[int] = None) -> KnowledgeBase:
    """The KnowledgeBase for a search: `KnowledgeBase.from_config` with the run's flags applied.

    `--no-rerank` turns reranking off; `--reranker KEY` replaces the configured protocol;
    `--rerank` (strict) makes an unavailable reranker an error instead of a warning.
    """
    strict = getattr(args, "rerank", False)
    if getattr(args, "no_rerank", False):
        rtype = "none"
    else:
        rtype = (getattr(args, "reranker", None) or config.reranker_type or "none").lower()
    if strict and rtype == "none":
        print(f"Error: --rerank set but no reranker chosen. Use --reranker "
              f"{'|'.join(sorted(RERANKER_TYPES))} or set `reranker:` in the config.",
              file=sys.stderr)
        sys.exit(1)
    try:
        return KnowledgeBase.from_config(
            config, model=getattr(args, "model", None), threads=threads,
            reranker=rtype, reranker_model=getattr(args, "reranker_model", None),
            strict_reranker=strict,
            on_warning=lambda m: print(f"Warning: {m}", file=sys.stderr),
        )
    except Exception as e:
        if strict and rtype != "none":
            print(f"Error: reranker '{rtype}' unavailable: {e}", file=sys.stderr)
            sys.exit(1)
        raise


def _confirm_mass_change_on_tty(detail) -> bool:
    """The interactive half of the mass-change guard.

    This lives in the CLI on purpose: a library that calls input() hangs or crashes
    in any process without a usable stdin. The engine decides nothing here — it
    hands us the numbers and we answer.
    """
    print(f"\n⚠  {detail}", file=sys.stderr)
    print("   This often means the source was corrupted, moved, or re-pointed — "
          "not a normal edit.", file=sys.stderr)
    if not sys.stdin.isatty():
        print("   Refusing to re-embed unattended. Re-run with --yes to accept, "
              "--force to rebuild, or --no-reindex-guard to skip this check.", file=sys.stderr)
        return False
    try:
        return input("   Re-index anyway? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _load_sources(config: Config, source_arg: str,
                  content_type: Optional[str] = None) -> list[DataSourceBase]:
    """Resolve --source into DataSource objects. 'list' prints and exits."""
    if source_arg == "list":
        print_sources(config)
        sys.exit(0)
    try:
        return resolve_sources(config, source_arg, content_type)
    except UnknownSource as e:
        print(f"{e} Use --source list to see them.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _resolve_throttle(args: argparse.Namespace, config: Config) -> tuple[Optional[float], str, int, int]:
    """Resolve throttle settings: CLI flag > config. Bare --throttle fills sensible
    defaults (half cores + low priority) for anything not otherwise set."""
    cores = getattr(args, "cores_fraction", None)
    if cores is None:
        cores = config.throttle_cores
    priority = getattr(args, "priority", None) or config.throttle_priority
    pause_ms = getattr(args, "pause_ms", None)
    pause_ms = config.throttle_pause_ms if pause_ms is None else pause_ms
    pause_every = getattr(args, "pause_every", None)
    pause_every = config.throttle_pause_every if pause_every is None else pause_every

    if getattr(args, "throttle", False):
        if cores is None:
            cores = 0.5
        if priority == "normal" and getattr(args, "priority", None) is None:
            priority = "low"
    return cores, priority, pause_ms, pause_every


def cmd_index(args: argparse.Namespace, config: Config) -> None:
    sources = _load_sources(config, getattr(args, "source", "all"))
    _, chunk_size, overlap, min_chunk = _effective(args, config)

    if getattr(args, "preview", False):
        _preview_chunks(_scan_kb(args, config), sources, config, chunk_size, overlap, min_chunk, args)
        return

    cores, priority, pause_ms, pause_every = _resolve_throttle(args, config)
    threads = cores_to_threads(cores)
    if threads or priority == "low" or pause_ms:
        print(f"[throttle] cores={cores if cores else 'all'} (threads={threads or 'default'})  "
              f"priority={priority}  pause={pause_ms}ms/{pause_every} files", file=sys.stderr)
    if priority == "low":
        lower_process_priority()

    # Mass-change guard: config toggle, minus a per-run override; threshold from flag or config.
    guard = config.reindex_guard and not getattr(args, "no_reindex_guard", False)
    guard_threshold = getattr(args, "reindex_threshold", None)
    if guard_threshold is None:
        guard_threshold = config.reindex_guard_threshold

    as_json = getattr(args, "json", False)
    kb = _kb_for(args, config, search=False, threads=threads)
    switch = getattr(args, "switch_model", False)
    if switch:
        # A switch re-embeds everything; `--source` would leave the other sources' vectors
        # in a different space, so the run always covers every configured source.
        sources = _load_sources(config, "all")
    results = kb.index_many(
        sources, chunk_size=chunk_size, overlap=overlap, min_chunk=min_chunk,
        force=args.force, switch_model=switch,
        limit=getattr(args, "limit", None), limit_per_source=getattr(args, "limit_per_source", False),
        pause_ms=pause_ms, pause_every=pause_every,
        guard=guard, guard_threshold=guard_threshold,
        assume_yes=getattr(args, "yes", False),
        # No progress on stdout in --json mode; it would break the document.
        on_progress=None if as_json else lambda m: print(m, flush=True),
        on_confirm=_confirm_mass_change_on_tty,
    )

    if as_json:
        emit_json(results)

    # An aborted guard used to be indistinguishable from success. Make it an exit code.
    if any(r.aborted for r in results):
        sys.exit(1)


def _preview_chunks(kb: KnowledgeBase, sources: list[DataSourceBase], config: Config,
                    chunk_size: int, overlap: int, min_chunk: int,
                    args: argparse.Namespace) -> None:
    """Write `KnowledgeBase.preview` output for each source to a file. Nothing is embedded."""
    file_filter = getattr(args, "file", None)
    max_chars = getattr(args, "max_chars", 0)
    out_arg = getattr(args, "out", None)

    source_ids = "-".join(s.source_id for s in sources)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if out_arg:
        out_path = Path(out_arg)
    else:
        tmp_dir = config.base_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        out_path = tmp_dir / f"kb-preview-{source_ids}-{ts}.txt"

    total_files = total_chunks = 0
    # Same budget semantics as a real index run: --limit caps the whole preview.
    limit = getattr(args, "limit", None)
    per_source = getattr(args, "limit_per_source", False)
    remaining = limit

    with out_path.open("w", encoding="utf-8") as fh:
        def p(*a, **kw):
            kw.setdefault("file", fh)
            print(*a, **kw)

        for source in sources:
            if limit is not None and not per_source and remaining <= 0:
                break
            source_limit = None if limit is None else (limit if per_source else remaining)
            files = kb.preview(source, chunk_size, overlap, min_chunk, limit=source_limit, file_name=file_filter)
            if file_filter and not files:
                print(f"[{source.source_id}] No file named '{file_filter}' found.", file=sys.stderr)
                continue
            if limit is not None and not per_source:
                remaining -= len(files)

            p(f"\n{'='*60}")
            p(f"Source: {source.source_id}  |  {len(files)} file(s)  |  chunk_size={chunk_size}")
            p(f"{'='*60}")

            for pf in files:
                if pf.skipped:
                    p(f"\n[{pf.rel_path}] — empty/unparseable, skipped")
                    continue
                total_files += 1
                total_chunks += len(pf.chunks)

                p(f"\n{'─'*60}")
                p(f"FILE: {pf.rel_path}  ({pf.body_chars} chars body → {len(pf.chunks)} chunks)")
                p(f"{'─'*60}")
                for i, c in enumerate(pf.chunks):
                    body_preview = c.text if not max_chars else c.text[:max_chars]
                    suffix = "…" if max_chars and len(c.text) > max_chars else ""
                    p(f"\n  Chunk {i+1}/{len(pf.chunks)}  ({len(c.text)} chars)")
                    p(f"  breadcrumb: {c.breadcrumb or '(none)'}")
                    p()
                    for line in (body_preview + suffix).splitlines():
                        p(f"    {line}")

        p(f"\n{'='*60}")
        p(f"Total: {total_files} file(s), {total_chunks} chunks")
        p(f"{'='*60}")

    print(f"Preview written to: {out_path}")


def _search_defaults(args: argparse.Namespace, config: Config) -> tuple[int, bool, int, Optional[str], bool]:
    """Resolve search params: CLI flag > `search:` config block > built-in default."""
    n = args.n if args.n is not None else (config.search_n or DEFAULT_N)
    # separate/fused: --fused forces off, --separate forces on, else the config default.
    if getattr(args, "fused", False):
        separate = False
    elif getattr(args, "separate", False):
        separate = True
    else:
        separate = config.search_separate
    max_chars = args.max_chars if args.max_chars is not None else (config.search_max_chars or 0)
    content_type = getattr(args, "content_type", None) or config.search_content_type
    timing = getattr(args, "timing", False) or config.search_timing
    return n, separate, max_chars, content_type, timing


def cmd_search(args: argparse.Namespace, config: Config) -> None:
    n, separate, max_chars, content_type, timing = _search_defaults(args, config)
    sources = _load_sources(config, getattr(args, "source", "all"), content_type)
    if not args.queries:
        print("Error: at least one query is required.", file=sys.stderr)
        sys.exit(1)
    kb = _kb_for(args, config, search=True)
    common = dict(
        sources=sources,
        queries=args.queries,
        n=n,
        content_type_filter=content_type,
        rerank_candidates=getattr(args, "rerank_candidates", None),
        cand_multiplier=config.cand_multiplier,
        cand_min=config.cand_min,
        cand_max=config.cand_max,
        strict_rerank=getattr(args, "rerank", False),
        timing=timing,
    )

    # Batch mode: each query gets its OWN top-n block (no cross-query merging).
    as_json = getattr(args, "json", False)

    if separate:
        groups = kb.search_grouped(**common)
        if as_json:
            emit_json({"mode": "separate",
                       "groups": [{"query": q, "hits": h} for q, h in groups]})
        else:
            for q, hits in groups:
                print("#" * 60)
                print(f"# Query: {q}  ({len(hits)} results)")
                print("#" * 60 + "\n")
                if hits:
                    print_results(hits, max_chars)
                else:
                    print("No results.\n")
    else:
        # Fused mode (default): all queries merged into one ranked list.
        hits = kb.search(**common)
        if as_json:
            emit_json({"mode": "fused", "queries": args.queries, "hits": hits})
        elif not hits:
            print("No results.")
        else:
            if len(args.queries) > 1:
                print(f"[{len(args.queries)} queries merged into one ranked list]")
            print_results(hits, max_chars)

    # The nudge goes to stderr in every mode, so a --json caller (usually an agent, the
    # reader the nudge exists for) sees it without the JSON document being touched. A served
    # instance evaluated freshness itself and sent the notices along with the hits.
    notices = getattr(kb, "last_notices", None)
    if notices is None:
        notices = FreshnessTracker.from_config(config).evaluate(kb, sources)
    for message in notices:
        print(message, file=sys.stderr)


def cmd_status(args: argparse.Namespace, config: Config) -> None:
    sources = _load_sources(config, getattr(args, "source", "all"))
    kb = _kb_for(args, config, search=False)
    statuses = kb.status(sources)
    if getattr(args, "json", False):
        emit_json(statuses)
        return
    tracker = FreshnessTracker.from_config(config) if config.freshness_enabled else None
    for st in statuses:
        print_status(st, tracker.nudges_for(st.source_id) if tracker else None)


def cmd_info(args: argparse.Namespace, config: Config) -> None:
    """Describe the instance and its sources — what is in here, and how much.

    Deliberately separate from `status`: status answers "is the index current",
    info answers "what does this hold and is it worth querying". A caller picking
    a source reads the descriptions, which status has no reason to show.
    """
    sources = _load_sources(config, getattr(args, "source", "all"))
    kb = _kb_for(args, config, search=False)
    inf = kb.info(sources, name=config.name)
    if getattr(args, "json", False):
        emit_json(inf)
        return
    print_info(inf)


def _scan_kb(args: argparse.Namespace, config: Config, threads: Optional[int] = None) -> KnowledgeBase:
    """A KnowledgeBase for index/scan/info/watch: the reranker is never used there, so it
    is left out and no reranker model or API key is needed."""
    return KnowledgeBase.from_config(config, model=getattr(args, "model", None), threads=threads,
                                     reranker="none")


def cmd_scan(args: argparse.Namespace, config: Config) -> None:
    sources = _load_sources(config, getattr(args, "source", "all"))
    kb = _kb_for(args, config, search=False)
    results = [kb.scan(s) for s in sources]
    if getattr(args, "json", False):
        emit_json(results)
        return
    any_stale = False
    for res in results:
        if not res.tracked:
            print(f"Scan [{res.source_id}]: not tracked yet — run "
                  f"`basic_kb index --source {res.source_id}` once to enable change detection.")
            continue
        line = (f"Scan [{res.source_id}]: {res.files_on_disk} files on disk  |  "
                f"new={res.new} changed={res.updated} unchanged={res.unchanged} deleted={res.deleted}")
        if res.stale:
            any_stale = True
            print(f"{line}  ->  {res.stale} stale")
        else:
            print(f"{line}  ->  up to date")
    if any_stale:
        print("\nRe-index with: basic_kb index")


def cmd_watch(args: argparse.Namespace, config: Config) -> None:
    from .watcher import Watcher, resolve_settings

    sources = _load_sources(config, getattr(args, "source", "all"))
    raw_by_id = {s["id"]: s for s in config.sources}
    debounce_override = getattr(args, "debounce", None)

    watched = []
    for src in sources:
        settings = resolve_settings(raw_by_id.get(src.source_id, {}), debounce_override)
        if settings.enabled:
            watched.append((src, settings))
        else:
            print(f"  (skipping '{src.source_id}': watch disabled in config)", file=sys.stderr)
    if not watched:
        print("No sources have watching enabled.", file=sys.stderr)
        sys.exit(1)

    _, chunk_size, overlap, min_chunk = _effective(args, config)

    # Same throttle as `index`: the watcher is a long-lived background embedder, so the
    # config's cores_fraction / priority must bind here too or it will take every core.
    cores, priority, pause_ms, pause_every = _resolve_throttle(args, config)
    threads = cores_to_threads(cores)
    if threads or priority == "low":
        print(f"[throttle] cores={cores if cores else 'all'} (threads={threads or 'default'})  "
              f"priority={priority}", file=sys.stderr)
    if priority == "low":
        lower_process_priority()

    kb = _scan_kb(args, config, threads=threads)
    if config.log_file:
        print(f"(logging events to {config.log_file})", file=sys.stderr)
    Watcher(kb, watched, chunk_size, overlap, min_chunk,
            guard=config.reindex_guard, guard_threshold=config.reindex_guard_threshold,
            on_event=print_watch_event).run_forever()


def cmd_vacuum(args: argparse.Namespace, config: Config) -> None:
    """Compact the store file now. Auto-vacuum (config `vacuum:`) normally does this
    after writes; this is for a one-off after a big manual clean-up."""
    kb = _kb_for(args, config, search=False)
    res = kb.vacuum()
    if getattr(args, "json", False):
        emit_json(res)
        return
    print(f"{'Vacuumed' if res.vacuumed else 'Vacuum skipped (busy or no store)'}: {res.path}  "
          f"({res.size_bytes:,} bytes; live chunks={res.live}, deleted since={res.deleted_since_vacuum})")


def _tri_state(on: bool, off: bool) -> Optional[bool]:
    """--x / --no-x pairs: True, False, or None for "use the config"."""
    if off:
        return False
    return True if on else None


def cmd_serve(args: argparse.Namespace, config: Config) -> None:
    """Run the HTTP API (and optionally the watcher) for this instance in the foreground."""
    from .server import KBServer

    _, chunk_size, overlap, min_chunk = _effective(args, config)
    cores, priority, _, _ = _resolve_throttle(args, config)
    threads = cores_to_threads(cores)
    if priority == "low":
        lower_process_priority()
    try:
        server = KBServer(
            config,
            host=getattr(args, "host", None), port=getattr(args, "port", None),
            auth=_tri_state(getattr(args, "auth", False), getattr(args, "no_auth", False)),
            watch=_tri_state(getattr(args, "watch", False), getattr(args, "no_watch", False)),
            allow_unauthenticated=getattr(args, "allow_unauthenticated", False),
            model=getattr(args, "model", None), threads=threads,
            chunk_size=chunk_size, overlap=overlap, min_chunk=min_chunk,
            debounce=getattr(args, "debounce", None),
            on_event=lambda m: print(f"[serve] {m}", file=sys.stderr, flush=True),
            on_watch_event=print_watch_event,
            on_warning=lambda m: print(f"Warning: {m}", file=sys.stderr),
        )
    except ValueError as e:          # the bind rule: non-loopback host without auth
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    server.serve_forever()


def cmd_keys(args: argparse.Namespace, config: Config) -> None:
    """Mint, list and revoke API keys for this instance's served API (ADR 0002). Local only:
    the file lives in the store dir and a running server picks up changes by itself."""
    store = ApiKeyStore(config.store_dir)
    sub = args.keys_cmd
    if sub == "create":
        record, plaintext = store.create(args.name)
        if getattr(args, "json", False):
            emit_json({"id": record.id, "name": record.name, "prefix": record.prefix, "key": plaintext})
            return
        print(plaintext)
        print(f"API key '{record.name}' (id {record.id}) created. This is the only time it is shown; "
              f"store it where the consumer reads it (BASIC_KB_API_KEY). Revoke with: basic-kb keys revoke {record.id}",
              file=sys.stderr)
    elif sub == "list":
        keys = store.list()
        if getattr(args, "json", False):
            emit_json(keys)
            return
        if not keys:
            print(f"No API keys. Create one with: basic-kb keys create --name NAME   (file: {store.path})")
            return
        print(f"{'ID':<10} {'NAME':<20} {'PREFIX':<12} {'CREATED':<17} STATE")
        for k in keys:
            created = datetime.datetime.fromtimestamp(k.created_at).strftime("%Y-%m-%d %H:%M")
            state = "active" if k.active else "revoked " + datetime.datetime.fromtimestamp(k.revoked_at).strftime("%Y-%m-%d")
            print(f"{k.id:<10} {k.name:<20} {k.prefix:<12} {created:<17} {state}")
    elif sub == "revoke":
        record = store.revoke(args.ident)
        print(f"Revoked API key '{record.name}' (id {record.id}). A running server refuses it from now on.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="Instance config YAML. If omitted: $BASIC_KB_CONFIG, "
                             "else basic-kb.yaml found by walking up from the current dir.")
    parser.add_argument("--env-file", metavar="FILE",
                        help="Dotenv file to load (e.g. for JINA_API_KEY). Overrides config env_file.")


def _attach_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-attach", action="store_true",
                        help="Run locally even if a `basic-kb serve` for this instance is alive "
                             "(also: BASIC_KB_NO_ATTACH=1). Writes still refuse while a server holds the store.")
    parser.add_argument("--attach", metavar="URL", default=None,
                        help="Use the served instance at URL (another machine, say) instead of anything local. "
                             "Example: --attach https://kb.example.com")
    parser.add_argument("--api-key", metavar="KEY", default=None,
                        help="Bearer API key for --attach / a served instance with auth on "
                             "(default: $BASIC_KB_API_KEY, which the config's env_file may set).")


def _shared_args(parser: argparse.ArgumentParser) -> None:
    _config_args(parser)
    _attach_args(parser)
    known = ", ".join(FastEmbedEmbedder.SUPPORTED)
    parser.add_argument("--model", default=None, metavar="NAME",
                        help=f"Embedding model alias or HF id (default: from config). Known: {known}")
    parser.add_argument("--chunk-size", type=int, default=None, metavar="N",
                        help="Max chunk size in chars (default: from config)")
    parser.add_argument("--overlap", type=int, default=None, metavar="N",
                        help="Overlap between chunks in chars (default: from config)")
    parser.add_argument("--min-chunk", type=int, default=None, metavar="N",
                        help="Minimum chunk size to keep in chars (default: from config)")
    parser.add_argument("--source", default="all", metavar="ID",
                        help="Source id(s) to operate on, comma-separated; 'all' (default) or 'list'.")


def _rerank_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-rerank", action="store_true", help="Disable reranking entirely")
    group.add_argument("--rerank", action="store_true",
                       help="Strict mode: error instead of falling back if the reranker fails")
    # Choices track the registry, so registering a backend needs no CLI edit.
    parser.add_argument("--reranker", choices=sorted(RERANKER_TYPES) + ["none"], default=None,
                        help="Reranker backend, overriding the config "
                             "(local=on-device, everything else=cloud)")
    parser.add_argument("--reranker-model", default=None, metavar="MODEL",
                        help="Reranker model/alias for the chosen backend (default: per-backend)")
    parser.add_argument("--rerank-candidates", type=int, default=None, metavar="N",
                        help="Candidates to fetch before reranking (default: 3× --n, max 50)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="basic_kb",
        description="basic-kb — local semantic search over markdown/text sources",
        epilog=(
            "Run a command with -h for its full options, e.g.  basic_kb index -h\n\n"
            "Common usage:\n"
            "  basic_kb search \"a statement the note would contain\"   search (all sources)\n"
            "  basic_kb search \"angle one\" \"angle two\"                multi-query, merged (better recall)\n"
            "  basic_kb search \"topic a\" \"topic b\" --separate         batch: n results per query\n"
            "  basic_kb search --source list                          list configured sources\n"
            "  basic_kb index                                         incremental: new/changed only\n"
            "  basic_kb index --force                                 re-embed the selected sources (same model)\n"
            "  basic_kb index --switch-model                          embedding model changed: wipe + rebuild all\n"
            "  basic_kb index --limit 10                              embed only the first N files total (test)\n"
            "  basic_kb status                                        chunk/doc counts per source\n"
            "  basic_kb scan                                          new/changed/deleted files vs the index\n"
            "  basic_kb watch                                         auto-reindex edited files (foreground)\n"
            "  basic_kb vacuum                                        compact the store file now\n"
            "  basic_kb serve [--watch] [--auth]                      HTTP API for this instance (foreground)\n"
            "  basic_kb keys create --name NAME                       mint an API key for the served API\n"
            "  basic_kb --inspect                                     resolved settings (freshness template)\n\n"
            "Attach: when `serve` runs for this instance, every command above uses it instead of loading\n"
            "        models again; --no-attach runs locally, --attach URL [--api-key K] targets a remote one.\n\n"
            "Search flags:  --n N (results)  --separate (batch: n per query)  --max-chars N  --content-type T  --timing\n"
            "Reranking:     --reranker local|jina-compatible|deepinfra-compatible|none  --reranker-model M  --no-rerank  --rerank (strict)\n"
            "Index flags:   --force  --switch-model  --limit N [--limit-per-source]  --preview [--file NAME]  --yes  --no-reindex-guard\n"
            "Throttle:      --throttle  --cores-fraction F  --priority low|normal  --pause-ms MS [--pause-every N]\n"
            "Watch:         --debounce SEC (0=immediate; per-source `watch:` config otherwise)\n"
            "Tuning (any):  --model NAME  --chunk-size N  --overlap N  --min-chunk N\n"
            "Config:        --config FILE, else $BASIC_KB_CONFIG, else basic-kb.yaml up the tree."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # A subparser's defaults overwrite the parent's on the namespace, so `--config`
    # stays per-command; --inspect resolves it from $BASIC_KB_CONFIG or the walk-up.
    parser.add_argument("--inspect", action="store_true",
                        help="Print resolved runtime settings for this instance (currently the "
                             "freshness nudge template in force) and exit. Takes no command: "
                             "basic_kb --inspect")
    # Not required: a bare `basic_kb` (or `basic_kb help`) prints help instead of erroring.
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("help", help="Show this help (same as -h)")

    p_index = sub.add_parser("index", help="Embed and index documents")
    _shared_args(p_index)
    p_index.add_argument("--force", action="store_true",
                         help="Clear the selected sources and re-embed them from scratch (same model)")
    p_index.add_argument("--switch-model", action="store_true",
                         help="Accept that embedding_model/embedding changed: wipe the whole store and re-embed "
                              "every source (implies --force and --source all). Without it a model mismatch "
                              "refuses to index anything.")
    p_index.add_argument("--limit", type=int, default=None, metavar="N",
                         help="Index only the first N files in total across the selected sources, in "
                              "source order (test runs). Sources left over once the budget is spent are "
                              "skipped. Example: index --limit 10")
    p_index.add_argument("--limit-per-source", action="store_true",
                         help="Make --limit apply per source instead of to the run as a whole, so every "
                              "selected source gets its own N files. "
                              "Example: index --limit 5 --limit-per-source")
    p_index.add_argument("--preview", action="store_true",
                         help="Preview chunks without embedding (dry-run). "
                              "Example: index --source notes --preview --file some.md")
    p_index.add_argument("--file", metavar="FILENAME",
                         help="Limit --preview to a single file by name, e.g. some-page.md")
    p_index.add_argument("--max-chars", type=int, default=0, metavar="N",
                         help="Truncate chunk content in --preview output to N chars (default: 0 = full)")
    p_index.add_argument("--out", metavar="FILE",
                         help="Write --preview output to FILE instead of the auto tmp/ path.")
    p_index.add_argument("--throttle", action="store_true",
                         help="Ease CPU load while indexing: ~half the cores + low OS priority.")
    p_index.add_argument("--cores-fraction", type=float, default=None, metavar="F",
                         help="Fraction of CPU cores the embedder may use, e.g. 0.5 (overrides --throttle/config).")
    p_index.add_argument("--priority", choices=["low", "normal"], default=None,
                         help="OS process priority while indexing (default: config, else normal).")
    p_index.add_argument("--pause-ms", type=int, default=None, metavar="MS",
                         help="Sleep MS milliseconds every --pause-every embedded files (hard CPU duty cap).")
    p_index.add_argument("--pause-every", type=int, default=None, metavar="N",
                         help="Pause cadence in embedded files (used with --pause-ms; default 50).")
    p_index.add_argument("--json", action="store_true",
                         help="emit the IndexResult as JSON; suppresses progress output")
    p_index.add_argument("--yes", "-y", action="store_true",
                         help="Auto-accept the mass-change safety prompt (for unattended re-indexing).")
    p_index.add_argument("--no-reindex-guard", action="store_true",
                         help="Skip the mass-change corruption check for this run.")
    p_index.add_argument("--reindex-threshold", type=float, default=None, metavar="F",
                         help="Churn fraction 0-1 that triggers the mass-change prompt "
                              "(default: config, else 0.9). Example: --reindex-threshold 0.75")

    p_search = sub.add_parser("search", help="Search the index")
    _shared_args(p_search)
    _rerank_args(p_search)
    p_search.add_argument("queries", nargs="*",
                          help="One or more queries. Default: merged into one ranked list "
                               "(re-framings of one need). With --separate: one result set each.")
    mode = p_search.add_mutually_exclusive_group()
    mode.add_argument("--separate", "--batch", dest="separate", action="store_true",
                      help="Batch mode: return --n results per query in its own block, instead "
                           "of merging all queries into one list. Use when the queries ask for "
                           "different things. Example: search \"coffee gear\" \"tax deadlines\" --separate")
    mode.add_argument("--fused", "--merge", dest="fused", action="store_true",
                      help="Force fused mode (one merged ranked list), overriding a "
                           "`search.separate: true` default in the config.")
    p_search.add_argument("--n", type=int, default=None, metavar="N",
                          help=f"Number of results (default: {DEFAULT_N}, or `search.n` in the config). "
                               f"In --separate mode, per query.")
    p_search.add_argument("--content-type", default=None, metavar="TYPE",
                          help="Filter by frontmatter content_type (markdown sources). "
                               "Default: `search.content_type` in the config, else none.")
    p_search.add_argument("--max-chars", type=int, default=None, metavar="N",
                          help="Truncate each result to N chars (default: `search.max_chars`, else 0 = full)")
    p_search.add_argument("--json", action="store_true",
                          help="emit hits as JSON instead of formatted text")
    p_search.add_argument("--timing", action="store_true",
                          help="Print per-phase timings (embed/retrieve/rerank/total) to stderr. "
                               "Also enabled by `search.timing: true` in the config.")

    p_info = sub.add_parser("info", help="Describe the sources: what each holds, how big it is")
    _shared_args(p_info)
    p_info.add_argument("--json", action="store_true", help="emit the instance description as JSON")

    p_status = sub.add_parser("status", help="Show index stats")
    p_status.add_argument("--json", action="store_true", help="emit per-source status as JSON")
    _shared_args(p_status)

    p_scan = sub.add_parser("scan", help="Check staleness: new/changed/deleted files vs the index (no embedding)")
    p_scan.add_argument("--json", action="store_true", help="emit the scan diff as JSON")
    _shared_args(p_scan)

    p_vacuum = sub.add_parser("vacuum", help="Compact the store file now (auto-vacuum normally handles this)")
    p_vacuum.add_argument("--json", action="store_true", help="emit before/after stats as JSON")
    _shared_args(p_vacuum)

    p_serve = sub.add_parser("serve", help="Serve this instance over HTTP (and optionally watch); foreground")
    _shared_args(p_serve)
    p_serve.add_argument("--host", default=None, metavar="ADDR",
                         help="Bind address (default: `serve.host` in the config, else 127.0.0.1). A non-loopback "
                              "address needs --auth or --allow-unauthenticated.")
    p_serve.add_argument("--port", type=int, default=None, metavar="N",
                         help="Port (default: `serve.port`, else 8765; 0 = pick a free one)")
    auth = p_serve.add_mutually_exclusive_group()
    auth.add_argument("--auth", action="store_true", help="Require a bearer API key on every route (keys: `basic-kb keys`)")
    auth.add_argument("--no-auth", action="store_true", help="Serve without authentication (default unless `serve.auth: true`)")
    watch = p_serve.add_mutually_exclusive_group()
    watch.add_argument("--watch", action="store_true", help="Also run the file watcher inside the server process")
    watch.add_argument("--no-watch", action="store_true", help="Serve without the watcher (default unless `serve.watch: true`)")
    p_serve.add_argument("--allow-unauthenticated", action="store_true",
                         help="Acknowledge serving on a non-loopback host with auth off; for setups where a proxy, "
                              "tailnet or tunnel in front of basic-kb handles access control.")
    p_serve.add_argument("--debounce", type=int, default=None, metavar="SEC",
                         help="Watcher debounce override, as for `watch`.")
    p_serve.add_argument("--throttle", action="store_true", help="Ease CPU load: ~half the cores + low OS priority.")
    p_serve.add_argument("--cores-fraction", type=float, default=None, metavar="F",
                         help="Fraction of CPU cores the embedder may use, e.g. 0.5.")
    p_serve.add_argument("--priority", choices=["low", "normal"], default=None,
                         help="OS process priority (default: config, else normal).")

    p_keys = sub.add_parser("keys", help="Manage API keys for the served API (create, list, revoke); local only")
    ksub = p_keys.add_subparsers(dest="keys_cmd", required=True)
    k_create = ksub.add_parser("create", help="Mint a key; the plaintext is printed once")
    _config_args(k_create)
    k_create.add_argument("--name", required=True, metavar="NAME", help="Who or what will use it, e.g. autotemple")
    k_create.add_argument("--json", action="store_true", help="emit {id, name, prefix, key} as JSON")
    k_list = ksub.add_parser("list", help="List keys (hashes only; never the plaintext)")
    _config_args(k_list)
    k_list.add_argument("--json", action="store_true", help="emit the key records as JSON")
    k_revoke = ksub.add_parser("revoke", help="Revoke a key by id (or by name when unambiguous)")
    _config_args(k_revoke)
    k_revoke.add_argument("ident", metavar="ID_OR_NAME")

    p_watch = sub.add_parser("watch",
                             help="Watch sources and auto-reindex edited files (foreground; Ctrl-C to stop)")
    _shared_args(p_watch)
    p_watch.add_argument("--debounce", type=int, default=None, metavar="SEC",
                         help="Reindex a file after it's been quiet this long, overriding each source's "
                              "config for this run (default 30s; 0 = reindex immediately). Example: --debounce 300")
    p_watch.add_argument("--throttle", action="store_true",
                         help="Ease CPU load while reindexing: ~half the cores + low OS priority.")
    p_watch.add_argument("--cores-fraction", type=float, default=None, metavar="F",
                         help="Fraction of CPU cores the embedder may use, e.g. 0.5 (overrides --throttle/config).")
    p_watch.add_argument("--priority", choices=["low", "normal"], default=None,
                         help="OS process priority while reindexing (default: config, else normal).")

    return parser


def _utf8_streams() -> None:
    """Windows consoles default to cp1252; our output uses →, ─, ⚠. Force UTF-8 so
    printing results never raises UnicodeEncodeError. (No-op where already UTF-8.)
    A CLI concern: a host application that imports the library keeps its own streams."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bare invocation or `help` → print top-level help and exit (no config needed).
    if args.cmd in (None, "help") and not args.inspect:
        parser.print_help()
        return

    # Resolve the config: explicit flag > $BASIC_KB_CONFIG > basic-kb.yaml up the tree.
    config_path = Path(getattr(args, "config", None)).expanduser() if getattr(args, "config", None) else find_config()
    if config_path is None:
        print(
            "No config found. Do one of:\n"
            "  • pass --config PATH\n"
            "  • set BASIC_KB_CONFIG=PATH\n"
            "  • add a basic-kb.yaml to this directory (or any parent).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load secrets first: --env-file (explicit) wins over config env_file (both setdefault).
    if getattr(args, "env_file", None):
        load_env_file(Path(args.env_file).expanduser())
    config = load_config(config_path)
    if config.env_file and config.env_file.exists():
        load_env_file(config.env_file)
    if config.log_file:
        setup_file_logging(config.log_file, config.log_level,
                           config.log_max_bytes, config.log_backup_count)

    if args.inspect:
        print_inspect(config)
        return

    try:
        {"index": cmd_index, "search": cmd_search, "status": cmd_status,
         "scan": cmd_scan, "watch": cmd_watch, "info": cmd_info, "vacuum": cmd_vacuum,
         "serve": cmd_serve, "keys": cmd_keys}[args.cmd](args, config)
    except BasicKBError as e:
        # Library errors are deliberate and already say what to do; a traceback adds nothing.
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
