"""basic-kb command-line interface.

  python -m basic_kb index  --config CFG [--source ID|all] [--force] [--preview]
  python -m basic_kb search "..." --config CFG [--source ID|all] [--n N]
  python -m basic_kb status --config CFG [--source ID|all]

Config is resolved automatically: --config flag > $BASIC_KB_CONFIG > a basic-kb.yaml
found by walking up from the current directory (see README).
Multi-query search improves recall:
  python -m basic_kb search "price too high" "budget concern" --config CFG

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
import sys
from pathlib import Path
from typing import Optional

from .config import Config, find_config, load_config, load_env_file
from .core import DEFAULT_N, KnowledgeBase
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


def _build_kb(args: argparse.Namespace, config: Config) -> KnowledgeBase:
    model, *_ = _effective(args, config)
    embedder = FastEmbedEmbedder(alias=model)

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

def cmd_index(args: argparse.Namespace, config: Config) -> None:
    sources = _load_sources(config, getattr(args, "source", "all"))
    _, chunk_size, overlap, min_chunk = _effective(args, config)

    if getattr(args, "preview", False):
        _preview_chunks(sources, config, chunk_size, overlap, min_chunk, args)
        return

    kb = _build_kb(args, config)
    for source in sources:
        kb.index(source=source, chunk_size=chunk_size, overlap=overlap,
                 min_chunk=min_chunk, force=args.force, limit=getattr(args, "limit", None))


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

    hits = kb.search(
        sources=sources,
        queries=args.queries,
        n=args.n,
        content_type_filter=content_type,
        rerank_candidates=getattr(args, "rerank_candidates", None),
        strict_rerank=getattr(args, "rerank", False),
        timing=getattr(args, "timing", False) or config.timing,
    )
    if not hits:
        print("No results.")
        return
    if len(args.queries) > 1:
        print(f"[{len(args.queries)} queries merged]")
    _print_results(hits, args.max_chars)


def cmd_status(args: argparse.Namespace, config: Config) -> None:
    sources = _load_sources(config, getattr(args, "source", "all"))
    kb = _build_kb(args, config)
    kb.status(sources)


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
            "  basic_kb search \"angle one\" \"angle two\"                multi-query, better recall\n"
            "  basic_kb search --source list                          list configured sources\n"
            "  basic_kb index                                         incremental: new/changed only\n"
            "  basic_kb index --force                                 rebuild the whole index\n"
            "  basic_kb index --limit 10                              embed only the first N files (test)\n"
            "  basic_kb status                                        chunk/doc counts per source\n\n"
            "Search flags:  --n N (results)  --max-chars N (truncate)  --content-type T  --timing\n"
            "Reranking:     --reranker local|jina|none  --reranker-model M  --no-rerank  --rerank (strict)\n"
            "Index flags:   --force  --limit N  --preview [--file NAME]\n"
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

    p_search = sub.add_parser("search", help="Search the index")
    _shared_args(p_search)
    _rerank_args(p_search)
    p_search.add_argument("queries", nargs="*",
                          help="One or more queries (multiple are merged for better recall).")
    p_search.add_argument("--n", type=int, default=DEFAULT_N, metavar="N",
                          help=f"Number of results (default: {DEFAULT_N})")
    p_search.add_argument("--content-type", default=None, metavar="TYPE",
                          help="Filter by frontmatter content_type (markdown sources).")
    p_search.add_argument("--max-chars", type=int, default=0, metavar="N",
                          help="Truncate each result to N chars (default: 0 = full)")
    p_search.add_argument("--timing", action="store_true",
                          help="Print per-phase timings (embed/retrieve/rerank/total) to stderr. "
                               "Also enabled by `timing: true` in the config.")

    p_status = sub.add_parser("status", help="Show index stats")
    _shared_args(p_status)

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

    {"index": cmd_index, "search": cmd_search, "status": cmd_status}[args.cmd](args, config)


if __name__ == "__main__":
    main()
