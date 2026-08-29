"""KnowledgeBase — orchestrates indexing and semantic search over DataSources.

Storage is SqliteVecStore (store.py): one SQLite file per instance holding vectors,
chunk text, metadata and the per-file manifest. Search is exact cosine top-k. Why
not an HNSW store: docs/adr/0001.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .chunkers import DEFAULT_CHUNK_SIZE, DEFAULT_MIN_CHUNK, DEFAULT_OVERLAP
from .embedders import EmbedderBase
from .errors import IndexNotFound, MassChangeRefused, QueryFailed, StoreError
from .models import FileError, IndexResult, InstanceInfo, SearchResult, SourceInfo, SourceStatus
from .rerankers import RerankerBase
from .sources import DataSourceBase
from .store import SqliteVecStore, VacuumPolicy

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
        logger.warning("could not lower process priority: %s", e)


def cores_to_threads(fraction: Optional[float]) -> Optional[int]:
    """Map a fraction of CPU cores (e.g. 0.5) to a thread count. None → None (use all)."""
    if not fraction:
        return None
    # os.cpu_count() reports host cores and ignores cgroup limits, so in a container
    # it sizes the thread pool against hardware the process cannot use. Prefer the
    # affinity mask where the platform has one.
    try:
        cpu = len(os.sched_getaffinity(0))
    except AttributeError:               # macOS, Windows — no affinity mask
        cpu = os.cpu_count() or 1
    return max(1, round(fraction * cpu))


@dataclass
class ScanResult:
    """Read-only diff of a source's files on disk vs. what was last indexed."""
    source_id: str
    label: str
    tracked: bool        # False if nothing indexed yet for this source
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

    All sources share one SQLite file; `source_id` partitions it. Searching multiple
    sources merges and re-ranks results across them.
    """

    def __init__(
        self,
        embedder: EmbedderBase,
        store_dir: Path,
        reranker: Optional[RerankerBase] = None,
        vacuum: Optional[VacuumPolicy] = None,
    ) -> None:
        self.embedder = embedder
        self.store_dir = Path(store_dir)
        self.store = SqliteVecStore(self.store_dir, vacuum=vacuum)
        self.reranker = reranker
        # Serialises writes within this process: SQLite allows one writer at a time and
        # index()/reindex_paths() read-then-write. Reentrant: index() holds it while
        # calling helpers that take it too.
        # NOTE: process-local. Two processes on one store still need external
        # coordination — see README.
        self._write_lock = threading.RLock()

    # --- TEMPORARY: legacy Chroma store detection --------------------------------------
    def legacy_chroma_leftovers(self) -> list[Path]:
        """Pre-ADR-0001 ChromaDB files still present in store_dir (empty list = none).

        TEMPORARY (2026-08-29): remove together with SqliteVecStore.legacy_chroma_leftovers
        once every instance on every machine has been rebuilt on sqlite-vec.
        """
        found = self.store.legacy_chroma_leftovers()
        if found:
            logger.warning("legacy ChromaDB store found in %s (%d item(s)); rebuild with `basic_kb index --force`",
                           self.store_dir, len(found))
        return found

    def prepare_model_switch(self, sources: list[DataSourceBase], accept: bool = False) -> Optional[str]:
        """Call before a (re)index run. If the store was built with a different embedding model
        than this KnowledgeBase uses, refuse to index anything — a model switch is never a
        side effect of a normal run or of `force`. Only with `accept=True` (CLI:
        `--switch-model`) and every indexed source in `sources` is the store emptied and the
        switch logged; the returned note says what was cleared. None when models match."""
        info = self.store.model_info()
        if not info or info[0] == self.embedder.model_id:
            return None
        indexed = set(self.store.sources())
        requested = {s.source_id for s in sources}
        missing = sorted(indexed - requested)
        if not accept:
            raise StoreError(
                f"Model switch detected: the store at {self.store.path} holds vectors from {info[0]!r} "
                f"(dim {info[1]}) but this run embeds with {self.embedder.model_id!r}. Nothing was indexed. "
                f"If this is intended, run `basic_kb index --switch-model` — it wipes the store and "
                f"re-embeds every source. Otherwise restore the previous embedding_model/embedding block.")
        if missing:
            raise StoreError(
                f"Model switch needs every indexed source in the run; missing: {', '.join(missing)}. "
                f"Run `basic_kb index --switch-model` without --source (all sources are rebuilt).")
        with self._write_lock:
            n = self.store.clear_all()
        msg = f"Model switch {info[0]!r} -> {self.embedder.model_id!r}: cleared {n} chunks across {sorted(indexed)}."
        logger.warning(msg)
        return msg

    @staticmethod
    def _rel_path(source: DataSourceBase, f: Path) -> str:
        try:
            return f.relative_to(source.directory).as_posix()
        except ValueError:
            return f.name

    def scan(self, source: DataSourceBase) -> ScanResult:
        """Diff the source's files on disk against the stored manifest. Read-only; no embedding."""
        manifest = self.store.manifest(source.source_id)
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

    def _guard_allows(
        self, source: DataSourceBase, changed: int, deleted: int,
        base: int, frac: float, threshold: float, assume_yes: bool,
        on_confirm: Optional[Callable[[MassChangeRefused], bool]],
    ) -> bool:
        """Decide whether to proceed when a source's files mostly changed at once.

        Such a jump usually means the source was corrupted, moved, or re-pointed —
        not a normal edit. The library never prompts: it accepts when `assume_yes`,
        otherwise asks `on_confirm` if one was supplied, otherwise refuses. The CLI
        passes an `on_confirm` that prompts on a TTY; a server passes none and gets
        a MassChangeRefused it can turn into a 409.
        """
        detail = MassChangeRefused(
            source_id=source.source_id, changed=changed, deleted=deleted,
            base=base, fraction=frac, threshold=threshold,
        )
        if assume_yes:
            logger.warning("guard auto-accepted: source=%s changed=%d deleted=%d frac=%.2f",
                           source.source_id, changed, deleted, frac)
            return True
        if on_confirm is not None:
            return bool(on_confirm(detail))
        logger.warning("guard refused: source=%s changed=%d deleted=%d frac=%.2f",
                       source.source_id, changed, deleted, frac)
        return False

    @staticmethod
    def _emit(on_progress: Optional[Callable[[str], None]], msg: str) -> None:
        """Progress goes to the caller if it asked for it, and to the log always.

        The library does not print. The CLI passes `on_progress=print`; a server
        passes nothing and reads the returned IndexResult instead.
        """
        logger.info("%s", msg)
        if on_progress is not None:
            on_progress(msg)

    @staticmethod
    def _chunk_ids(rel_path: str, chunks) -> list[str]:
        """Content-derived chunk ids: `<rel_path>::<sha1(text)[:16]>`, with `#n` appended for
        repeated identical text within one file. Unchanged text keeps its id (and vector) across
        re-indexes, so appending to a file re-embeds only the new chunk(s). The positional index
        stays available as `chunk.id_suffix` in the metadata for ordering."""
        ids, seen = [], {}
        for c in chunks:
            h = hashlib.sha1(c.text.encode("utf-8")).hexdigest()[:16]
            n = seen.get(h, 0)
            seen[h] = n + 1
            ids.append(f"{rel_path}::{h}" + (f"#{n}" if n else ""))
        return ids

    # Chunks per embedding wave. A wave collects the *new* chunks of many files and embeds
    # them in one call, so an API backend's concurrency spans files instead of being
    # bounded by the 1-9 chunks a typical note has (one request per file = one round-trip
    # of latency per file). 4096 = 64 requests x 64 texts at the default settings; a crash
    # mid-wave loses at most this much work, since files are written after the wave.
    WAVE_CHUNKS = 4096

    def _write_file(self, source: DataSourceBase, rel_path: str, file_hash: str, chunks,
                    vectors: Optional[dict[str, list[float]]] = None) -> tuple[int, int]:
        """Sync one file's chunks into the store, embedding only chunks whose text is new.
        `vectors` (text -> vector) supplies precomputed embeddings from a wave; anything not
        in it is embedded on the spot. Returns (chunks_embedded, chunks_reused). Empty
        `chunks` drops the file's rows but records its hash, so a short file stays tracked."""
        ids = self._chunk_ids(rel_path, chunks)
        docs = [c.text for c in chunks]
        metas = [{**c.metadata, "rel_path": rel_path, "content_hash": file_hash, "position": c.id_suffix}
                 for c in chunks]

        def embed(texts: list[str]) -> list[list[float]]:
            have = vectors or {}
            missing = [t for t in texts if t not in have]
            if missing:
                have = {**have, **dict(zip(missing, self.embedder.embed(missing)))}
            return [have[t] for t in texts]

        return self.store.sync_file(source.source_id, rel_path, file_hash, ids, docs, metas,
                                    embed=embed, model_id=self.embedder.model_id)

    def _embed_wave(self, source: DataSourceBase, prepared: list[tuple[str, list]]) -> dict[str, list[float]]:
        """Embed, in one call, every chunk text in `prepared` [(rel_path, chunks)] that the
        store does not already hold. Returns text -> vector for the caller to hand to
        _write_file. Duplicate texts across files are embedded once."""
        todo: dict[str, None] = {}
        for rel_path, chunks in prepared:
            have = self.store.existing_chunk_ids(source.source_id, rel_path)
            for cid, ch in zip(self._chunk_ids(rel_path, chunks), chunks):
                if cid not in have:
                    todo[ch.text] = None
        texts = list(todo)
        if not texts:
            return {}
        return dict(zip(texts, self.embedder.embed(texts)))

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
        on_progress: Optional[Callable[[str], None]] = None,
        on_confirm: Optional[Callable[[MassChangeRefused], bool]] = None,
    ) -> IndexResult:
        """Index a DataSource into the store.

        Returns an IndexResult describing what happened — including `aborted=True`
        when the mass-change guard stopped the run, and `errors` listing any files
        that could not be parsed. Nothing is printed; pass `on_progress` for a live
        feed and `on_confirm` to be asked about a mass change.

        `limit` caps the number of files (first N, stable order). `pause_ms` and
        `pause_every` throttle CPU duty between embedded files.

        Each file is committed as it completes, so an interrupted run resumes rather
        than starting over.
        """
        with self._write_lock:
            return self._index_locked(
                source, chunk_size, overlap, min_chunk, force, limit, pause_ms,
                pause_every, guard, guard_threshold, assume_yes, on_progress, on_confirm,
            )

    def _index_locked(
        self, source, chunk_size, overlap, min_chunk, force, limit, pause_ms,
        pause_every, guard, guard_threshold, assume_yes, on_progress, on_confirm,
    ) -> IndexResult:
        files = source.get_files()
        result = IndexResult(
            source_id=source.source_id, label=source.label,
            files_on_disk=len(files), limited_to=limit,
        )
        if not files:
            result.abort_reason = f"No files found for source {source.label!r}."
            logger.warning("index: no files for source=%s", source.source_id)
            self._emit(on_progress, result.abort_reason)
            return result

        if limit is not None:
            files = files[:limit]
            self._emit(on_progress, f"[limit] indexing only the first {len(files)} file(s).")

        self.store.check_model(self.embedder.model_id)   # a model switch must go through prepare_model_switch()
        if force:
            n = self.store.clear_source(source.source_id)
            if n:
                self._emit(on_progress, f"Cleared existing index for {source.source_id!r} ({n} chunks).")

        # Pre-scan: hash every file once and decide the real work up front.
        prev_manifest: dict = {} if force else self.store.manifest(source.source_id)
        entries = [(f, self._rel_path(source, f), _file_hash(f)) for f in files]
        seen: dict[str, str] = {rp: h for _, rp, h in entries}
        to_process = [(f, rp, h) for f, rp, h in entries if prev_manifest.get(rp) != h]
        deleted_files = [rp for rp in prev_manifest if rp not in seen] if limit is None else []
        work_total = len(to_process)
        result.unchanged = len(entries) - work_total

        # Corruption guard: a large fraction of previously-indexed files changing or
        # vanishing at once is more likely a broken source than a real bulk edit.
        if guard and not force and prev_manifest:
            changed_existing = sum(1 for _, rp, _ in to_process if rp in prev_manifest)
            churn = changed_existing + len(deleted_files)
            base = len(prev_manifest)
            frac = churn / base if base else 0.0
            if churn >= _GUARD_MIN_FILES and frac >= guard_threshold:
                if not self._guard_allows(source, changed_existing, len(deleted_files),
                                          base, frac, guard_threshold, assume_yes, on_confirm):
                    result.aborted = True
                    result.abort_reason = str(MassChangeRefused(
                        source.source_id, changed_existing, len(deleted_files),
                        base, frac, guard_threshold))
                    self._emit(on_progress, result.abort_reason)
                    return result

        chunker = source.make_chunker(chunk_size, overlap, min_chunk)
        prune_note = f", {len(deleted_files)} to prune" if deleted_files else ""
        self._emit(on_progress, (
            f"Indexing {source.label}: {len(entries)} files on disk, "
            f"{work_total} to (re)embed, {result.unchanged} unchanged{prune_note}  "
            f"(model={self.embedder.model_id}, chunker={chunker.name}, "
            f"chunk={chunk_size}, overlap={overlap})"
        ))
        if work_total == 0 and not deleted_files:
            self._emit(on_progress, "  Already up to date — nothing to embed.")

        was_indexed_files = set(self.store.chunk_ids_by_file(source.source_id))
        since_pause = 0
        processed = 0
        # Files are parsed and hash-diffed a wave at a time, the wave's new chunks are
        # embedded in ONE call (concurrent for API backends), then each file is written as
        # its own committed transaction — a crash keeps every file written so far.
        i = 0
        while i < len(to_process):
            prepared: list[tuple[str, list, str, Path]] = []
            wave_chunks = 0
            while i < len(to_process) and (wave_chunks < self.WAVE_CHUNKS or not prepared):
                f, rel_path, file_hash = to_process[i]
                i += 1
                processed += 1
                try:
                    doc = source.parse_file(f)
                    chunks = chunker.chunk(doc) if doc is not None else []
                except Exception as e:
                    # One unreadable file must not abandon a run that has already
                    # embedded hundreds. Record it and carry on; the manifest simply
                    # does not learn this file, so the next run retries it.
                    logger.exception("index: failed to parse %s/%s", source.source_id, rel_path)
                    result.errors.append(FileError(rel_path=rel_path, error=f"{type(e).__name__}: {e}"))
                    self._emit(on_progress, f"  [{processed}/{work_total}] SKIPPED {rel_path} — {e}")
                    continue
                prepared.append((rel_path, chunks, file_hash, f))
                wave_chunks += len(chunks)

            vectors = self._embed_wave(source, [(rp, ch) for rp, ch, _, _ in prepared])

            for rel_path, chunks, file_hash, _ in prepared:
                was_indexed = rel_path in was_indexed_files
                if not chunks:
                    result.empty += 1
                    self._write_file(source, rel_path, file_hash, [])   # tracked, holds no chunks
                    self._emit(on_progress, f"  {rel_path}  (no chunks — tracked, nothing to embed)")
                    continue
                n_new, n_kept = self._write_file(source, rel_path, file_hash, chunks, vectors)
                result.chunks_embedded += n_new
                result.chunks_reused += n_kept
                kept = f", {n_kept} unchanged" if n_kept else ""
                self._emit(on_progress, f"  {rel_path}  ({n_new} chunks embedded{kept})")
                result.updated += 1 if was_indexed else 0
                result.added += 0 if was_indexed else 1

                if pause_ms > 0 and pause_every > 0:
                    since_pause += 1
                    if since_pause >= pause_every:
                        time.sleep(pause_ms / 1000.0)
                        since_pause = 0
            self._emit(on_progress, f"  [{processed}/{work_total}] files done")

        # Prune files that no longer exist on disk. Full runs only — a --limit run
        # sees a subset, so absent files there are not actually deleted.
        if limit is None and deleted_files:
            result.pruned = self.store.remove_files(source.source_id, deleted_files)

        result.total_chunks = self.store.count(source.source_id)
        self._emit(on_progress, (
            f"\nDone [{source.source_id}].\n"
            f"  files:  new={result.added} updated={result.updated} "
            f"unchanged={result.unchanged} empty={result.empty} pruned={result.pruned}"
            f"  ({result.embedded} files touched this run)\n"
            f"  chunks: {result.chunks_embedded} embedded, {result.chunks_reused} reused, "
            f"{result.total_chunks} total in index"
        ))
        if result.errors:
            self._emit(on_progress, f"  {len(result.errors)} file(s) failed — see result.errors")
        logger.info("index done: source=%s new=%d updated=%d unchanged=%d empty=%d "
                    "pruned=%d chunks=%d errors=%d",
                    source.source_id, result.added, result.updated, result.unchanged,
                    result.empty, result.pruned, result.total_chunks, len(result.errors))
        return result

    def stale_paths(self, source: DataSourceBase) -> list[Path]:
        """Files that differ from the manifest (new/changed on disk + deleted).

        Deleted files are returned as their (now non-existent) path so reindex_paths
        can prune them. Used for the watcher's startup reconcile of offline edits.
        """
        manifest = self.store.manifest(source.source_id)
        live = {self._rel_path(source, f): f for f in source.get_files()}
        changed = [f for rp, f in live.items() if manifest.get(rp) != _file_hash(f)]
        deleted = [source.directory / rp for rp in manifest if rp not in live]
        return changed + deleted

    def reindex_paths(
        self,
        source: DataSourceBase,
        paths,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        min_chunk: int = DEFAULT_MIN_CHUNK,
    ) -> dict:
        """Reindex a specific set of files, not the whole source (the watcher's write path).

        Embeds created/changed files, drops chunks for files that became empty, prunes
        deleted files, and updates just those manifest entries. Returns a counts dict.
        Callers must serialize this (one writer).
        """
        with self._write_lock:
            return self._reindex_paths_locked(source, paths, chunk_size, overlap, min_chunk)

    def _reindex_paths_locked(self, source, paths, chunk_size, overlap, min_chunk) -> dict:
        rels: dict[str, Path] = {}
        for p in paths:
            p = Path(p)
            rels[self._rel_path(source, p)] = p
        counts = {"embedded": 0, "empty": 0, "pruned": 0, "unchanged": 0,
                  "chunks_embedded": 0, "chunks_reused": 0}
        if not rels:
            return counts

        chunker = source.make_chunker(chunk_size, overlap, min_chunk)
        manifest = self.store.manifest(source.source_id)
        gone = [rp for rp, p in rels.items() if not p.exists()]
        if gone:
            counts["pruned"] = self.store.remove_files(source.source_id, gone)
            for rp in gone:
                logger.info("reindex_paths: pruned %s/%s", source.source_id, rp)

        prepared: list[tuple[str, list, str]] = []
        for rp, p in rels.items():
            if not p.exists():
                continue
            file_hash = _file_hash(p)
            # Unchanged content already in the index: nothing to do. Guards against
            # mtime-only touches and spurious watcher events.
            if manifest.get(rp) == file_hash:
                counts["unchanged"] += 1
                logger.debug("reindex_paths: unchanged %s/%s", source.source_id, rp)
                continue
            doc = source.parse_file(p)
            prepared.append((rp, chunker.chunk(doc) if doc is not None else [], file_hash))

        vectors = self._embed_wave(source, [(rp, ch) for rp, ch, _ in prepared])
        for rp, chunks, file_hash in prepared:
            if not chunks:
                self._write_file(source, rp, file_hash, [])
                counts["empty"] += 1
                continue
            n_new, n_kept = self._write_file(source, rp, file_hash, chunks, vectors)
            counts["embedded"] += 1
            counts["chunks_embedded"] += n_new
            counts["chunks_reused"] += n_kept
            logger.info("reindex_paths: %s/%s — %d chunks embedded, %d reused",
                        source.source_id, rp, n_new, n_kept)
        return counts

    def vacuum(self) -> bool:
        """Compact the store file now, regardless of the auto-vacuum policy."""
        with self._write_lock:
            return self.store.vacuum(reason="requested")

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
        """Fused multi-query search: all queries pool into ONE ranked list of n hits.

        Each query is run against every source; hits are merged by score (highest wins
        per chunk) and optionally reranked. Best when the queries are re-framings of the
        same information need (they reinforce recall). For n hits *per* query instead,
        use search_grouped(). With timing=True, prints per-phase durations.
        """
        self._require_store()
        return self._run_query_group(
            sources, queries, n, content_type_filter, rerank_candidates,
            cand_multiplier, cand_min, cand_max, strict_rerank, timing,
        )

    def search_grouped(
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
    ) -> list[tuple[str, list[SearchResult]]]:
        """Batch search (msearch-style): each query independently returns its OWN top-n.

        Returns [(query, hits), ...] — no cross-query merging. Best when the queries ask
        for *different* things in one call and you want a full result set for each.
        """
        self._require_store()
        out: list[tuple[str, list[SearchResult]]] = []
        for q in queries:
            hits = self._run_query_group(
                sources, [q], n, content_type_filter, rerank_candidates,
                cand_multiplier, cand_min, cand_max, strict_rerank, timing, timing_label=q,
            )
            out.append((q, hits))
        return out

    def _require_store(self) -> None:
        if not self.store.exists():
            raise IndexNotFound(
                f"No index at {self.store.path}. Run: python -m basic_kb index"
            )

    def _run_query_group(
        self, sources: list[DataSourceBase], queries: list[str], n: int,
        content_type_filter: Optional[str], rerank_candidates: Optional[int],
        cand_multiplier: int, cand_min: int, cand_max: int,
        strict_rerank: bool, timing: bool, timing_label: str = "",
    ) -> list[SearchResult]:
        """Run one group of queries into a single ranked list (embed→retrieve→merge→rerank).

        The reranker scores against queries[0], so a group is one information need.
        """
        t0 = time.perf_counter()
        t_embed = t_retrieve = t_rerank = 0.0
        best: dict[str, SearchResult] = {}
        missing: list[str] = []      # sources with nothing indexed yet
        resolved = 0                 # sources that actually got queried

        if self.reranker and rerank_candidates is None:
            # How many top hits to rerank: clamp(n × multiplier, min, max).
            # If min exceeds max (misconfig), min wins.
            hi = max(cand_max, cand_min)
            rerank_candidates = min(max(n * cand_multiplier, cand_min), hi)
        fetch_n = (rerank_candidates or n) if self.reranker else n

        self.store.check_model(self.embedder.model_id)       # wrong model = confident nonsense; refuse
        # Embed each query once; the same vector serves every source.
        _te = time.perf_counter()
        q_embs = [self.embedder.query_embed([q])[0] for q in queries]
        t_embed += time.perf_counter() - _te

        for source in sources:
            if not self.store.source_indexed(source.source_id):
                # One un-indexed source among several is a normal, recoverable state.
                # Every source missing is not — that is caught after the loop.
                logger.warning("search: no index for source=%s", source.source_id)
                missing.append(source.source_id)
                continue
            resolved += 1

            for q_emb in q_embs:
                try:
                    _tr = time.perf_counter()
                    rows = self.store.knn(source.source_id, q_emb, fetch_n, content_type_filter)
                    t_retrieve += time.perf_counter() - _tr
                except Exception as e:
                    # Swallowing this returned an empty list indistinguishable from
                    # "nothing matched" — a broken store looked like a healthy one.
                    raise QueryFailed(
                        f"Query failed on source {source.source_id!r}: {e}"
                    ) from e

                for row in rows:
                    full_id = f"{source.source_id}::{row.chunk_id}"
                    score = 1.0 - row.distance
                    if full_id not in best or score > best[full_id].score:
                        best[full_id] = SearchResult(doc=row.doc, metadata=row.metadata, score=score)

        if resolved == 0:
            raise IndexNotFound(
                "None of the requested sources are indexed "
                f"({', '.join(missing) or 'no sources given'}). "
                "An empty result would be indistinguishable from no matches."
            )

        hits = sorted(best.values(), key=lambda r: r.score, reverse=True)

        _trr = time.perf_counter()
        if self.reranker and hits:
            candidates = hits[: rerank_candidates or n]
            try:
                hits = self.reranker.rerank(queries[0], candidates, top_n=n)
            except Exception as e:
                if strict_rerank:
                    raise
                logger.warning("reranking failed, falling back to cosine scores: %s", e)
                hits = hits[:n]
        else:
            hits = hits[:n]
        t_rerank = time.perf_counter() - _trr

        if timing:
            label = f" [{timing_label}]" if timing_label else ""
            logger.info(
                "timing (ms)%s: embed=%.0f retrieve=%.0f rerank=%.0f total=%.0f "
                "[cold process — embed & rerank include one-time model load]",
                label, t_embed * 1000, t_retrieve * 1000, t_rerank * 1000,
                (time.perf_counter() - t0) * 1000,
            )

        logger.info(
            "search: sources=%s queries=%r n=%d candidates=%s hits=%d "
            "embed_ms=%d retrieve_ms=%d rerank_ms=%d reranker=%s",
            [s.source_id for s in sources], queries, n, rerank_candidates, len(hits),
            t_embed * 1000, t_retrieve * 1000, t_rerank * 1000,
            type(self.reranker).__name__ if self.reranker else None,
        )
        return hits

    def info(self, sources: list[DataSourceBase], name: str = "") -> InstanceInfo:
        """Describe the knowledge base: what each source holds and how big it is.

        Written for a reader — usually an agent — deciding WHICH source to query.
        That is a different question from `status`, which answers whether the index
        is current, so this carries each source's configured description and leaves
        staleness out.

        A source with no index yet comes back with chunks=0 and indexed=False rather
        than raising; an un-indexed source among several is a normal state.
        """
        out: list[SourceInfo] = []
        for source in sources:
            chunks = self.store.count(source.source_id)
            out.append(SourceInfo(
                source_id=source.source_id,
                label=source.label,
                description=source.description,
                type=type(source).type_name or type(source).__name__,
                chunker=source.chunker_name,
                files=len(source.get_files()),
                chunks=chunks,
                indexed=chunks > 0,
            ))

        return InstanceInfo(
            name=name,
            model_id=self.embedder.model_id,
            store_dir=str(self.store_dir),
            sources=out,
        )

    def status(self, sources: list[DataSourceBase]) -> list[SourceStatus]:
        """Index stats per source, as data. Nothing is printed — the CLI formats it.

        Raises IndexNotFound if the store does not exist at all; a source that is
        merely un-indexed comes back with `indexed=False` rather than an exception,
        because that is a normal state in a multi-source instance.
        """
        self._require_store()
        out: list[SourceStatus] = []

        for source in sources:
            st = SourceStatus(
                source_id=source.source_id,
                label=source.label,
                directory=str(source.directory),
                store_dir=str(self.store_dir),
                model_id=self.embedder.model_id,
                directory_exists=source.directory.exists(),
                indexed=False,
            )

            if not self.store.source_indexed(source.source_id):
                logger.info("status: nothing indexed for source=%s", source.source_id)
                out.append(st)
                continue

            st.indexed = True
            st.chunks = self.store.count(source.source_id)
            st.chars = self.store.chars(source.source_id)

            res = self.scan(source)
            st.files_on_disk = res.files_on_disk
            st.tracked = res.tracked
            st.new, st.updated, st.deleted = res.new, res.updated, res.deleted

            if st.chunks == 0:
                out.append(st)
                continue

            all_metas = self.store.all_metadata(source.source_id)
            st.docs_with_chunks = len({m.get("rel_path") or m.get("file") for m in all_metas})

            dates = sorted(
                m.get("date", "unknown") for m in all_metas
                if m.get("date", "unknown") != "unknown"
            )
            if dates:
                st.date_min, st.date_max = dates[0], dates[-1]

            for m in all_metas:
                ct = str(m.get("content_type", "unknown"))
                if ct != "unknown":
                    st.content_types[ct] = st.content_types.get(ct, 0) + 1

            st.oversized_chunks = sum(1 for m in all_metas if m.get("oversized"))
            out.append(st)

        return out
