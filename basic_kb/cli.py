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

Reranking (search only; on by default when JINA_API_KEY is set):
  --no-rerank | --rerank (strict)  --reranker-model M  --rerank-candidates N

Secrets: set JINA_API_KEY in the environment, pass --env-file PATH, or set
`env_file:` in the config (CLI/shell win over the config file).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Optional

from .config import Config, find_config, load_config, load_env_file
from .core import DEFAULT_N, KnowledgeBase, cores_to_threads, lower_process_priority, setup_file_logging
from .embedders import FastEmbedEmbedder
from .models import SearchResult
from .rerankers import RerankerBase, build_reranker
from .sources import DataSourceBase, build_source


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


def _build_kb(args: argparse.Namespace, config: Config, threads: Optional[int] = None) -> KnowledgeBase:
    model, *_ = _effective(args, config)
    embedder = FastEmbedEmbedder(alias=model, threads=threads)

    reranker: Optional[RerankerBase] = None
    if not getattr(args, "no_rerank", False):
        # Type/model: CLI flag wins, else config, else off.
        rtype = (getattr(args, "reranker", None) or config.reranker_type or "none").lower()
        rmodel = getattr(args, "reranker_model", None) or config.reranker_model
        strict = getattr(args, "rerank", False)
        if rtype != "none":
            try:
                reranker = build_reranker(rtype, rmodel)
            except Exception as e:
                # e.g. jina selected but no API key. Strict → fail; else cosine-only.
                if strict:
                    print(f"Error: reranker '{rtype}' unavailable: {e}", file=sys.stderr)
                    sys.exit(1)
                print(f"Warning: reranker '{rtype}' unavailable, using cosine scores only ({e})",
                      file=sys.stderr)
        elif strict:
            print("Error: --rerank set but no reranker chosen. Use --reranker local|jina "
                  "or set `reranker:` in the config.", file=sys.stderr)
            sys.exit(1)

    return KnowledgeBase(embedder=embedder, chroma_dir=config.store_dir, reranker=reranker)


def _print_sources(config: Config) -> None:
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


def _load_sources(config: Config, source_arg: str,
                  content_type: Optional[str] = None) -> list[DataSourceBase]:
    """Resolve --source into DataSource objects. 'list' prints and exits."""
    configured = {s["id"]: s for s in config.sources}
    if source_arg == "list":
        _print_sources(config)
        sys.exit(0)
    ids = list(configured) if source_arg == "all" else [s.strip() for s in source_arg.split(",") if s.strip()]
    out: list[DataSourceBase] = []
    for sid in ids:
        if sid not in configured:
            print(f"Unknown source {sid!r}. Configured: {', '.join(configured)} (or 'all', 'list').",
                  file=sys.stderr)
            sys.exit(1)
        out.append(build_source(configured[sid], config.base_dir, content_type))
    return out


def _print_results(hits: list[SearchResult], max_chars: int) -> None:
    reranked = any(r.rerank_score is not None for r in hits)
    print(f"Top {len(hits)} results{'  [reranked]' if reranked else ''}\n")

    for rank, r in enumerate(hits, 1):
        meta = r.metadata
        score_str = f"score={round(r.score, 3)}"
        if r.rerank_score is not None:
            score_str += f"  rerank={round(r.rerank_score, 4)}"

        # Web-style metadata (url/content_type) gets a richer header; else date.
        if meta.get("url") or meta.get("content_type", "unknown") != "unknown":
            header = (
                f"[{rank}] {meta.get('title', '?')}  "
                f"[{meta.get('content_type', '?')}]  {score_str}"
            )
            ref = meta.get("url") or meta.get("file", "?")
        else:
            # Show the date only when there is one (transcripts have it; notes usually don't).
            date = meta.get("date")
            date_str = f"  ({date})" if date and date != "unknown" else ""
            header = f"[{rank}] {meta.get('title', '?')}{date_str}  {score_str}"
            ref = meta.get("file", "?")

        print("=" * 60)
        print(header)
        print(f"    {ref}")
        print("─" * 60)
        output = r.doc if not max_chars else r.doc[:max_chars]
        print(output)
        if max_chars and len(r.doc) > max_chars:
            print(f"  [...{len(r.doc) - max_chars} more chars]")
        print()


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
        _preview_chunks(sources, config, chunk_size, overlap, min_chunk, args)
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

    kb = _build_kb(args, config, threads=threads)
    for source in sources:
        kb.index(source=source, chunk_size=chunk_size, overlap=overlap,
                 min_chunk=min_chunk, force=args.force, limit=getattr(args, "limit", None),
                 pause_ms=pause_ms, pause_every=pause_every,
                 guard=guard, guard_threshold=guard_threshold,
                 assume_yes=getattr(args, "yes", False))


def _preview_chunks(sources: list[DataSourceBase], config: Config,
                    chunk_size: int, overlap: int, min_chunk: int,
                    args: argparse.Namespace) -> None:
    """Write chunks for each source to a file without embedding anything."""
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

    with out_path.open("w", encoding="utf-8") as fh:
        def p(*a, **kw):
            kw.setdefault("file", fh)
            print(*a, **kw)

        for source in sources:
            chunker = source.make_chunker(chunk_size, overlap, min_chunk)
            files = source.get_files()
            if file_filter:
                files = [f for f in files if f.name == file_filter]
                if not files:
                    print(f"[{source.source_id}] No file named '{file_filter}' found.", file=sys.stderr)
                    continue
            limit = getattr(args, "limit", None)
            if limit is not None:
                files = files[:limit]

            p(f"\n{'='*60}")
            p(f"Source: {source.source_id}  |  {len(files)} file(s)  |  chunk_size={chunk_size}")
            p(f"{'='*60}")

            for f in files:
                doc = source.parse_file(f)
                if doc is None:
                    p(f"\n[{f.name}] — empty/unparseable, skipped")
                    continue
                chunks = chunker.chunk(doc)
                total_files += 1
                total_chunks += len(chunks)

                p(f"\n{'─'*60}")
                p(f"FILE: {f.name}  ({len(doc.body)} chars body → {len(chunks)} chunks)")
                p(f"{'─'*60}")
                for i, c in enumerate(chunks):
                    body_preview = c.text if not max_chars else c.text[:max_chars]
                    suffix = "…" if max_chars and len(c.text) > max_chars else ""
                    p(f"\n  Chunk {i+1}/{len(chunks)}  ({len(c.text)} chars)")
                    p(f"  breadcrumb: {c.metadata.get('breadcrumb', '(none)')}")
                    p()
                    for line in (body_preview + suffix).splitlines():
                        p(f"    {line}")

        p(f"\n{'='*60}")
        p(f"Total: {total_files} file(s), {total_chunks} chunks")
        p(f"{'='*60}")

    print(f"Preview written to: {out_path}")


def cmd_search(args: argparse.Namespace, config: Config) -> None:
    content_type = getattr(args, "content_type", None)
    sources = _load_sources(config, getattr(args, "source", "all"), content_type)
    if not args.queries:
        print("Error: at least one query is required.", file=sys.stderr)
        sys.exit(1)
    kb = _build_kb(args, config)
    common = dict(
        sources=sources,
        queries=args.queries,
        n=args.n,
        content_type_filter=content_type,
        rerank_candidates=getattr(args, "rerank_candidates", None),
        cand_multiplier=config.cand_multiplier,
        cand_min=config.cand_min,
        cand_max=config.cand_max,
        strict_rerank=getattr(args, "rerank", False),
        timing=getattr(args, "timing", False) or config.timing,
    )

    # Batch mode: each query gets its OWN top-n block (no cross-query merging).
    if getattr(args, "separate", False):
        groups = kb.search_grouped(**common)
        for q, hits in groups:
            print("#" * 60)
            print(f"# Query: {q}  ({len(hits)} results)")
            print("#" * 60 + "\n")
            if hits:
                _print_results(hits, args.max_chars)
            else:
                print("No results.\n")
        _freshness_reminder(kb, sources, config)
        return

    # Fused mode (default): all queries merged into one ranked list.
    hits = kb.search(**common)
    if not hits:
        print("No results.")
        return
    if len(args.queries) > 1:
        print(f"[{len(args.queries)} queries merged into one ranked list]")
    _print_results(hits, args.max_chars)
    _freshness_reminder(kb, sources, config)


def cmd_status(args: argparse.Namespace, config: Config) -> None:
    sources = _load_sources(config, getattr(args, "source", "all"))
    kb = _build_kb(args, config)
    kb.status(sources)


def _scan_kb(args: argparse.Namespace, config: Config) -> KnowledgeBase:
    """A KnowledgeBase for scan/freshness — no reranker needed (scan only hashes files)."""
    model, *_ = _effective(args, config)
    return KnowledgeBase(embedder=FastEmbedEmbedder(alias=model), chroma_dir=config.store_dir)


def cmd_scan(args: argparse.Namespace, config: Config) -> None:
    sources = _load_sources(config, getattr(args, "source", "all"))
    kb = _scan_kb(args, config)
    any_stale = False
    for source in sources:
        res = kb.scan(source)
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
    from .watcher import resolve_settings, run_watch

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
    kb = _scan_kb(args, config)  # embedder only; reindex needs no reranker
    if config.log_file:
        print(f"(logging events to {config.log_file})", file=sys.stderr)
    run_watch(kb, watched, config, chunk_size, overlap, min_chunk)


def _format_freshness(template: str, res, days: int) -> str:
    fields = dict(source=res.source_id, new=res.new, updated=res.updated,
                  deleted=res.deleted, unchanged=res.unchanged, stale=res.stale,
                  total=res.files_on_disk, days=days)
    try:
        return template.format(**fields)
    except (KeyError, IndexError) as e:
        return (f"[basic-kb] freshness message has an invalid placeholder {e}; "
                f"valid: {', '.join(fields)}. Source '{res.source_id}' is stale "
                f"({res.stale} file(s)).")


def _freshness_reminder(kb: KnowledgeBase, sources: list[DataSourceBase], config: Config) -> None:
    """After a search, every `every_days`, check the queried sources and nudge if stale."""
    if not config.freshness_enabled:
        return
    state_path = config.store_dir / "freshness_state.json"
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}   # treat corrupt state as "never checked"; it gets rewritten below

    now = time.time()
    interval = max(0, config.freshness_every_days) * 86400
    due = [s for s in sources if now - float(state.get(s.source_id, 0)) >= interval]
    if not due:
        return

    messages: list[str] = []
    for s in due:
        res = kb.scan(s)
        state[s.source_id] = now   # reset the timer whether or not it was stale
        if res.tracked and res.stale:
            messages.append(_format_freshness(config.freshness_message, res, config.freshness_every_days))

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    for m in messages:
        print(m, file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="Instance config YAML. If omitted: $BASIC_KB_CONFIG, "
                             "else basic-kb.yaml found by walking up from the current dir.")
    parser.add_argument("--env-file", metavar="FILE",
                        help="Dotenv file to load (e.g. for JINA_API_KEY). Overrides config env_file.")
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
    parser.add_argument("--reranker", choices=["local", "jina", "none"], default=None,
                        help="Reranker backend, overriding the config (local=on-device, jina=cloud)")
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
            "  basic_kb index --force                                 rebuild the whole index\n"
            "  basic_kb index --limit 10                              embed only the first N files (test)\n"
            "  basic_kb status                                        chunk/doc counts per source\n"
            "  basic_kb scan                                          new/changed/deleted files vs the index\n"
            "  basic_kb watch                                         auto-reindex edited files (foreground)\n\n"
            "Search flags:  --n N (results)  --separate (batch: n per query)  --max-chars N  --content-type T  --timing\n"
            "Reranking:     --reranker local|jina|none  --reranker-model M  --no-rerank  --rerank (strict)\n"
            "Index flags:   --force  --limit N  --preview [--file NAME]  --yes  --no-reindex-guard\n"
            "Throttle:      --throttle  --cores-fraction F  --priority low|normal  --pause-ms MS [--pause-every N]\n"
            "Watch:         --debounce SEC (0=immediate; per-source `watch:` config otherwise)\n"
            "Tuning (any):  --model NAME  --chunk-size N  --overlap N  --min-chunk N\n"
            "Config:        --config FILE, else $BASIC_KB_CONFIG, else basic-kb.yaml up the tree."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Not required: a bare `basic_kb` (or `basic_kb help`) prints help instead of erroring.
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("help", help="Show this help (same as -h)")

    p_index = sub.add_parser("index", help="Embed and index documents")
    _shared_args(p_index)
    p_index.add_argument("--force", action="store_true",
                         help="Clear existing index and re-embed from scratch")
    p_index.add_argument("--limit", type=int, default=None, metavar="N",
                         help="Index only the first N files per source (test runs). "
                              "Example: index --limit 10")
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
    p_search.add_argument("--separate", "--batch", dest="separate", action="store_true",
                          help="Batch mode: return --n results per query in its own block, instead "
                               "of merging all queries into one list. Use when the queries ask for "
                               "different things. Example: search \"coffee gear\" \"tax deadlines\" --separate")
    p_search.add_argument("--n", type=int, default=DEFAULT_N, metavar="N",
                          help=f"Number of results (default: {DEFAULT_N}). In --separate mode, per query.")
    p_search.add_argument("--content-type", default=None, metavar="TYPE",
                          help="Filter by frontmatter content_type (markdown sources).")
    p_search.add_argument("--max-chars", type=int, default=0, metavar="N",
                          help="Truncate each result to N chars (default: 0 = full)")
    p_search.add_argument("--timing", action="store_true",
                          help="Print per-phase timings (embed/retrieve/rerank/total) to stderr. "
                               "Also enabled by `timing: true` in the config.")

    p_status = sub.add_parser("status", help="Show index stats")
    _shared_args(p_status)

    p_scan = sub.add_parser("scan", help="Check staleness: new/changed/deleted files vs the index (no embedding)")
    _shared_args(p_scan)

    p_watch = sub.add_parser("watch",
                             help="Watch sources and auto-reindex edited files (foreground; Ctrl-C to stop)")
    _shared_args(p_watch)
    p_watch.add_argument("--debounce", type=int, default=None, metavar="SEC",
                         help="Reindex a file after it's been quiet this long, overriding each source's "
                              "config for this run (default 30s; 0 = reindex immediately). Example: --debounce 300")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bare invocation or `help` → print top-level help and exit (no config needed).
    if args.cmd in (None, "help"):
        parser.print_help()
        return

    # Resolve the config: explicit flag > $BASIC_KB_CONFIG > basic-kb.yaml up the tree.
    config_path = Path(args.config).expanduser() if args.config else find_config()
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

    {"index": cmd_index, "search": cmd_search, "status": cmd_status,
     "scan": cmd_scan, "watch": cmd_watch}[args.cmd](args, config)


if __name__ == "__main__":
    main()
