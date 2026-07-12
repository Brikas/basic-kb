"""KnowledgeBase — orchestrates indexing and semantic search over DataSources."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chunkers import DEFAULT_CHUNK_SIZE, DEFAULT_MIN_CHUNK, DEFAULT_OVERLAP
from .embedders import EmbedderBase
from .models import SearchResult
from .rerankers import RerankerBase
from .sources import DataSourceBase

DEFAULT_N = 15

# Absolute floor for the mass-change guard: below this many changed/deleted files a
# high churn fraction is just a small source being edited, not corruption — don't nag.
_GUARD_MIN_FILES = 5

logger = logging.getLogger("basic_kb")


def setup_file_logging(path: Path, level: str = "INFO",
                       max_bytes: int = 10_000_000, backup_count: int = 20) -> None:
    """Attach a rotating file handler to the basic_kb logger so events go to `path`.

    Rotates at ~max_bytes (default 10 MB) keeping backup_count old files (default 20,
    ~200 MB of history) — logs neither grow unbounded nor get wiped. Idempotent per file.
    """
    from logging.handlers import RotatingFileHandler

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = str(path.resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "_bkb_target", None) == target:
            return  # already logging to this file
    handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    handler._bkb_target = target  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False


def _file_hash(path: Path) -> str:
    """Content fingerprint of a file (sha1 of its bytes). Identity only, not security."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def lower_process_priority() -> None:
    """Drop this process's OS scheduling priority so heavy work yields to foreground apps.
    Best-effort: warns (does not abort) if the platform call fails."""
    try:
        if sys.platform == "win32":
            import ctypes
            # restype/argtypes are required so the 64-bit pseudo-handle isn't
            # truncated to 32 bits (which makes SetPriorityClass fail).
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass.restype = ctypes.c_bool
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
                raise OSError(f"SetPriorityClass failed (err={ctypes.get_last_error()})")
        else:
            os.nice(10)  # raise niceness (lower priority)
    except Exception as e:
        print(f"Warning: could not lower process priority ({e}).", file=sys.stderr)


def cores_to_threads(fraction: Optional[float]) -> Optional[int]:
    """Map a fraction of CPU cores (e.g. 0.5) to a thread count. None → None (use all)."""
    if not fraction:
        return None
    cpu = os.cpu_count() or 1
    return max(1, round(fraction * cpu))


@dataclass
class ScanResult:
    """Read-only diff of a source's files on disk vs. what was last indexed."""
    source_id: str
    label: str
    tracked: bool        # False if no manifest yet (never indexed under the new scheme)
    files_on_disk: int
    new: int
    updated: int
    unchanged: int
    deleted: int

    @property
    def stale(self) -> int:
        """Files that differ from the index (would change it on re-index)."""
        return self.new + self.updated + self.deleted


class KnowledgeBase:
    """
    Orchestrates indexing and semantic search across one or more DataSources.

    Each DataSource maps to its own ChromaDB collection (named by source_id).
    Searching multiple sources merges and re-ranks results across collections.
    """

    def __init__(
        self,
        embedder: EmbedderBase,
        chroma_dir: Path,
        reranker: Optional[RerankerBase] = None,
    ) -> None:
        self.embedder = embedder
        self.chroma_dir = chroma_dir
        self.reranker = reranker

    def _client(self):
        import chromadb
        return chromadb.PersistentClient(path=str(self.chroma_dir))

    # --- Manifest: every file seen at index time -> content hash, per source. -----
    # Lets `scan` diff disk against the index without re-parsing, and correctly
    # treats files too short to chunk as "seen" (not perpetually "new").
    def _manifest_path(self) -> Path:
        return self.chroma_dir / "manifest.json"

    def _load_manifest(self) -> dict:
        p = self._manifest_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"Warning: manifest at {p} is corrupt ({e}); treating as empty.", file=sys.stderr)
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _rel_path(source: DataSourceBase, f: Path) -> str:
        try:
            return f.relative_to(source.directory).as_posix()
        except ValueError:
            return f.name

    def scan(self, source: DataSourceBase) -> ScanResult:
        """Diff the source's files on disk against the manifest. Read-only; no embedding."""
        manifest = self._load_manifest().get(source.source_id, {})
        tracked = bool(manifest)
        live = {self._rel_path(source, f): _file_hash(f) for f in source.get_files()}

        new = updated = unchanged = 0
        for rp, h in live.items():
            if rp not in manifest:
                new += 1
            elif manifest[rp] != h:
                updated += 1
            else:
                unchanged += 1
        deleted = sum(1 for rp in manifest if rp not in live)
        logger.info("scan: source=%s tracked=%s on_disk=%d new=%d updated=%d unchanged=%d deleted=%d",
                    source.source_id, tracked, len(live), new, updated, unchanged, deleted)
        return ScanResult(
            source_id=source.source_id, label=source.label, tracked=tracked,
            files_on_disk=len(live), new=new, updated=updated,
            unchanged=unchanged, deleted=deleted,
        )

    def _get_collection(self, client, collection_name: str):
        """Get or create a named ChromaDB collection, clearing on EF conflict."""
        ef = self.embedder.as_chroma_ef()
        try:
            return client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError as exc:
            if "embedding function conflict" not in str(exc).lower():
                raise
            print(f"Note: embedding function changed for '{collection_name}'. Clearing old index.")
            client.delete_collection(collection_name)
            return client.create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )

    def _confirm_mass_change(
        self, source: DataSourceBase, changed: int, deleted: int,
        base: int, frac: float, threshold: float, assume_yes: bool,
    ) -> bool:
        """Guard prompt before re-embedding a source whose files mostly changed at once.

        Such a jump usually means the source was corrupted, moved, or its path
        re-pointed — not a normal edit. Returns True to proceed. `assume_yes` accepts
        automatically; otherwise ask on a TTY, and refuse (return False) when running
        unattended so a bad run can't silently trash a good index.
        """
        pct, tpct = round(frac * 100), round(threshold * 100)
        print(
            f"\n⚠  Mass change on '{source.source_id}': {changed} changed + {deleted} deleted "
            f"of {base} indexed files = {pct}% (guard threshold {tpct}%).",
            file=sys.stderr,
        )
        print("   This often means the source was corrupted, moved, or re-pointed — not a normal edit.",
              file=sys.stderr)
        print(f"   Directory: {source.directory}", file=sys.stderr)
        if assume_yes:
            print("   Proceeding anyway (--yes / auto-accept).", file=sys.stderr)
            logger.warning("guard auto-accepted: source=%s changed=%d deleted=%d frac=%.2f",
                           source.source_id, changed, deleted, frac)
            return True
        if not sys.stdin.isatty():
            print("   Refusing to re-embed unattended. Re-run with --yes to accept, --force to "
                  "rebuild, or --no-reindex-guard to skip this check.", file=sys.stderr)
            return False
        try:
            answer = input("   Re-index anyway? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        return answer in ("y", "yes")

    def index(
        self,
        source: DataSourceBase,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        min_chunk: int = DEFAULT_MIN_CHUNK,
        force: bool = False,
        limit: Optional[int] = None,
        pause_ms: int = 0,
        pause_every: int = 50,
        guard: bool = True,
        guard_threshold: float = 0.9,
        assume_yes: bool = False,
    ) -> None:
        """Index all files from a DataSource into its own ChromaDB collection.

        `limit` caps the number of files (first N, stable order) — handy for test
        runs before committing to a long full embed. `pause_ms`/`pause_every` throttle
        CPU duty by sleeping between batches of embedded files. When `guard` is on and
        a fraction >= `guard_threshold` of already-indexed files changed/vanished at
        once (likely corruption), confirm before proceeding unless `assume_yes`.
        """
        files = source.get_files()
        if not files:
            print(f"No files found for source '{source.label}'. Nothing to index.", file=sys.stderr)
            return
        if limit is not None:
            files = files[:limit]
            print(f"[limit] indexing only the first {len(files)} file(s) of this source.")

        client = self._client()
        if force:
            try:
                client.delete_collection(source.source_id)
                print(f"Cleared existing index for '{source.source_id}'.")
            except Exception:
                pass

        col = self._get_collection(client, source.source_id)

        # Snapshot what's already indexed, keyed by each file's relative path.
        # rel_path is the stable per-file identity used to prune/replace chunks.
        snap = col.get(include=["metadatas"])
        ids_by_file: dict[str, list[str]] = {}       # rel_path -> its chunk ids
        for cid, meta in zip(snap["ids"], snap["metadatas"]):
            # Fall back to legacy 'file' key for chunks indexed before rel_path existed.
            rp = meta.get("rel_path") or meta.get("file") or cid.split("::")[0]
            ids_by_file.setdefault(rp, []).append(cid)

        # Pre-scan: hash every file once and decide the real work up front. The manifest
        # (which records ALL files incl. empty) is the skip oracle — a file whose hash
        # still matches is untouched since last index, so it never re-parses. This makes
        # progress honest ([k/work] not [i/all-files]) and mass-change corruption catchable.
        prev_manifest: dict = {} if force else self._load_manifest().get(source.source_id, {})
        entries = [(f, self._rel_path(source, f), _file_hash(f)) for f in files]
        seen: dict[str, str] = {rp: h for _, rp, h in entries}  # manifest: every file, incl. empty
        to_process = [(f, rp, h) for f, rp, h in entries if prev_manifest.get(rp) != h]
        deleted_files = [rp for rp in prev_manifest if rp not in seen] if limit is None else []
        work_total = len(to_process)
        unchanged = len(entries) - work_total

        # Corruption guard: a large fraction of previously-indexed files changing or
        # vanishing at once is more likely a broken/moved source than a real bulk edit.
        if guard and not force and prev_manifest:
            changed_existing = sum(1 for _, rp, _ in to_process if rp in prev_manifest)
            churn = changed_existing + len(deleted_files)
            base = len(prev_manifest)
            frac = churn / base if base else 0.0
            if churn >= _GUARD_MIN_FILES and frac >= guard_threshold:
                if not self._confirm_mass_change(
                    source, changed_existing, len(deleted_files), base, frac, guard_threshold, assume_yes
                ):
                    print(f"Aborted index for '{source.source_id}' — index left unchanged.", file=sys.stderr)
                    logger.warning("index aborted by guard: source=%s churn=%d/%d frac=%.2f",
                                   source.source_id, churn, base, frac)
                    return

        chunker = source.make_chunker(chunk_size, overlap, min_chunk)
        prune_note = f", {len(deleted_files)} to prune" if deleted_files else ""
        print(
            f"Indexing {source.label}: {len(entries)} files on disk, "
            f"{work_total} to (re)embed, {unchanged} unchanged{prune_note}  "
            f"(model={self.embedder.model_id}, chunker={chunker.name}, "
            f"chunk={chunk_size}, overlap={overlap})"
        )
        if work_total == 0 and not deleted_files:
            print("  Already up to date — nothing to embed.")
        logger.info("index start: source=%s files=%d work=%d unchanged=%d deleted=%d model=%s force=%s limit=%s",
                    source.source_id, len(entries), work_total, unchanged,
                    len(deleted_files), self.embedder.model_id, force, limit)

        added = updated = empty = pruned = 0
        # Work total is known from the pre-scan; "?" only as an honest fallback if it
        # ever couldn't be determined (it always can here — kept for defensiveness).
        total_str = str(work_total) if work_total is not None else "?"
        since_pause = 0             # embedded files since the last throttle pause

        for processed, (f, rel_path, file_hash) in enumerate(to_process, 1):
            was_indexed = rel_path in ids_by_file
            doc = source.parse_file(f)
            chunks = chunker.chunk(doc) if doc is not None else []
            if not chunks:
                empty += 1
                if was_indexed:  # used to have content, now none → drop stale chunks
                    col.delete(ids=ids_by_file[rel_path])
                continue

            # Print right before embedding (the slow step) so a stall points at this file.
            verb = "re-embedding" if was_indexed else "embedding"
            print(f"  [{processed}/{total_str}] {verb} {rel_path}  ({len(chunks)} chunks)", flush=True)

            if was_indexed:  # changed file → remove old chunks before re-adding
                col.delete(ids=ids_by_file[rel_path])

            new_ids, new_docs, new_metas = [], [], []
            for chunk in chunks:
                new_ids.append(f"{rel_path}::{chunk.id_suffix}")
                new_docs.append(chunk.text)
                new_metas.append({**chunk.metadata, "rel_path": rel_path, "content_hash": file_hash})
            col.add(documents=new_docs, metadatas=new_metas, ids=new_ids)
            updated += 1 if was_indexed else 0
            added += 0 if was_indexed else 1

            # Throttle: brief sleep between batches of embedded files to free the CPU.
            if pause_ms > 0 and pause_every > 0:
                since_pause += 1
                if since_pause >= pause_every:
                    time.sleep(pause_ms / 1000.0)
                    since_pause = 0

        # Prune files that no longer exist on disk. Only on a full run — a --limit
        # run sees a subset, so absent files there are not actually deleted.
        if limit is None and deleted_files:
            stale_ids = [cid for rp in deleted_files if rp in ids_by_file for cid in ids_by_file[rp]]
            if stale_ids:
                col.delete(ids=stale_ids)
            pruned = len(deleted_files)

        # Persist the manifest. Full run replaces this source's entry (drops deleted
        # files); a --limit run merges, since it only saw a subset.
        manifest = self._load_manifest()
        if limit is None:
            manifest[source.source_id] = seen
        else:
            manifest[source.source_id] = {**manifest.get(source.source_id, {}), **seen}
        self._save_manifest(manifest)

        total = col.count()
        print(
            f"\nDone [{source.source_id}].\n"
            f"  files:  new={added} updated={updated} unchanged={unchanged} "
            f"empty={empty} pruned={pruned}  ({added + updated} embedded this run)\n"
            f"  chunks: {total} total in index  (files are split into chunks; one file = many chunks)"
        )
        logger.info("index done: source=%s new=%d updated=%d unchanged=%d empty=%d pruned=%d chunks=%d",
                    source.source_id, added, updated, unchanged, empty, pruned, total)

    def search(
        self,
        sources: list[DataSourceBase],
        queries: list[str],
        n: int = DEFAULT_N,
        content_type_filter: Optional[str] = None,
        rerank_candidates: Optional[int] = None,
        cand_multiplier: int = 3,
        cand_min: int = 50,
        cand_max: int = 200,
        strict_rerank: bool = False,
        timing: bool = False,
    ) -> list[SearchResult]:
        """
        Multi-source, multi-query semantic search.

        Each query is run against every source collection independently.
        Results are merged by score (highest wins per unique chunk ID),
        then optionally reranked. With timing=True, prints per-phase durations.
        """
        if not self.chroma_dir.exists():
            print("No index found. Run: python -m basic_kb index")
            return []

        t0 = time.perf_counter()
        t_embed = t_retrieve = t_rerank = 0.0
        client = self._client()
        best: dict[str, SearchResult] = {}

        if self.reranker and rerank_candidates is None:
            # How many top hits to rerank: clamp(n × multiplier, min, max).
            # If min exceeds max (misconfig), min wins.
            hi = max(cand_max, cand_min)
            rerank_candidates = min(max(n * cand_multiplier, cand_min), hi)
        fetch_n = (rerank_candidates or n) if self.reranker else n

        for source in sources:
            try:
                col = client.get_collection(
                    name=source.source_id,
                    embedding_function=self.embedder.as_chroma_ef(),
                )
            except Exception:
                print(
                    f"  Note: no index for '{source.source_id}' — "
                    f"run: python -m basic_kb index --source {source.source_id}",
                    file=sys.stderr,
                )
                continue

            where: Optional[dict] = None
            if content_type_filter:
                where = {"content_type": {"$eq": content_type_filter}}

            for q in queries:
                _te = time.perf_counter()
                q_emb = self.embedder.query_embed([q])
                t_embed += time.perf_counter() - _te
                kwargs: dict = {
                    "query_embeddings": q_emb,
                    "n_results": fetch_n,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    kwargs["where"] = where
                try:
                    _tr = time.perf_counter()
                    results = col.query(**kwargs)
                    t_retrieve += time.perf_counter() - _tr
                except Exception as e:
                    print(f"  Query error on '{source.source_id}': {e}", file=sys.stderr)
                    continue

                for doc_id, doc, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    full_id = f"{source.source_id}::{doc_id}"
                    score = 1.0 - dist
                    if full_id not in best or score > best[full_id].score:
                        best[full_id] = SearchResult(doc=doc, metadata=meta, score=score)

        hits = sorted(best.values(), key=lambda r: r.score, reverse=True)

        _trr = time.perf_counter()
        if self.reranker and hits:
            candidates = hits[: rerank_candidates or n]
            try:
                hits = self.reranker.rerank(queries[0], candidates, top_n=n)
            except Exception as e:
                if strict_rerank:
                    raise
                print(f"Warning: reranking failed, falling back to cosine scores ({e})", file=sys.stderr)
                hits = hits[:n]
        else:
            hits = hits[:n]
        t_rerank = time.perf_counter() - _trr

        if timing:
            print(
                f"timing (ms): embed={t_embed*1000:.0f}  retrieve={t_retrieve*1000:.0f}  "
                f"rerank={t_rerank*1000:.0f}  total={(time.perf_counter()-t0)*1000:.0f}"
                "   [cold process — embed & rerank include one-time model load]",
                file=sys.stderr,
            )

        logger.info(
            "search: sources=%s queries=%r n=%d candidates=%s hits=%d "
            "embed_ms=%d retrieve_ms=%d rerank_ms=%d reranker=%s",
            [s.source_id for s in sources], queries, n, rerank_candidates, len(hits),
            t_embed * 1000, t_retrieve * 1000, t_rerank * 1000,
            type(self.reranker).__name__ if self.reranker else None,
        )
        return hits

    def status(self, sources: list[DataSourceBase]) -> None:
        """Print index stats for one or more sources."""
        if not self.chroma_dir.exists():
            print("No index found. Run: python -m basic_kb index")
            return

        client = self._client()

        for source in sources:
            print(f"\n{'='*55}")
            print(f"Source  : {source.label}  ({source.source_id})")
            print(f"ChromaDB: {self.chroma_dir}")
            print(f"Model   : {self.embedder.model_id}")

            # Warn loudly if the source path itself is gone (moved dir, unmounted drive,
            # broken symlink). The index can still look healthy while pointing at nothing.
            if not source.directory.exists():
                print(f"  ⚠  SOURCE PATH MISSING: {source.directory}")
                print("     directory does not exist — moved, unmounted, or a broken symlink.")
                print("     Any indexed chunks are now orphaned; fix the path or re-point the source.")

            try:
                col = client.get_collection(
                    name=source.source_id,
                    embedding_function=self.embedder.as_chroma_ef(),
                )
            except Exception:
                print(f"  No index found. Run: python -m basic_kb index --source {source.source_id}")
                continue

            total_chunks = col.count()
            if total_chunks == 0:
                print(f"  Index empty. Run: python -m basic_kb index --source {source.source_id}")
                continue

            all_metas: list[dict] = col.get(include=["metadatas"])["metadatas"]
            docs_with_chunks = len({m["file"] for m in all_metas})
            # Use the manifest (via scan) for staleness — it tracks every processed
            # file incl. those too short to chunk, so empties aren't false "unindexed".
            res = self.scan(source)

            print(f"Chunks  : {total_chunks:,}")
            print(f"Docs    : {docs_with_chunks:,} produced chunks / {res.files_on_disk} files on disk")
            if res.files_on_disk == 0:
                # Distinguish "path gone" (warned above) from "path exists but empty".
                if not source.directory.exists():
                    print(f"  ⚠  0 files — source path is missing (see warning above); {docs_with_chunks} indexed docs orphaned.")
                else:
                    print(f"  ⚠  0 files at {source.directory}")
                    print(f"     directory exists but holds no matching files — the {docs_with_chunks} indexed docs are now stale.")
            elif not res.tracked:
                print(f"           no manifest yet — run `python -m basic_kb index --source {source.source_id}` "
                      f"once so empty files are tracked (until then {res.new} files show as new).")
            elif res.stale:
                parts = [f"{res.new} new", f"{res.updated} changed", f"{res.deleted} deleted"]
                print(f"           {', '.join(p for p, n in zip(parts, (res.new, res.updated, res.deleted)) if n)}"
                      f" since last index — run: python -m basic_kb index --source {source.source_id}")
            else:
                print("           up to date  (files too short to chunk are tracked, not counted as unindexed)")

            dates = sorted(
                m.get("date", "unknown") for m in all_metas
                if m.get("date", "unknown") != "unknown"
            )
            if dates:
                print(f"Dates   : {dates[0]} → {dates[-1]}")

            # Content-type breakdown — shown for any source whose chunks carry a
            # real content_type (e.g. markdown sources with frontmatter).
            has_content_type = any(
                str(m.get("content_type", "unknown")) != "unknown" for m in all_metas
            )
            if has_content_type:
                ct_counts: dict[str, int] = {}
                for m in all_metas:
                    ct = str(m.get("content_type", "unknown"))
                    ct_counts[ct] = ct_counts.get(ct, 0) + 1
                for ct, count in sorted(ct_counts.items()):
                    print(f"  {ct}: {count:,} chunks")

            oversized_count = sum(1 for m in all_metas if m.get("oversized"))
            if oversized_count:
                print(f"  ⚠  oversized chunks: {oversized_count}")
