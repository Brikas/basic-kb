"""KnowledgeBase — orchestrates indexing and semantic search over DataSources."""
from __future__ import annotations

import hashlib
import json
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
from .errors import IndexNotFound, ManifestCorrupt, MassChangeRefused, QueryFailed
from .models import FileError, IndexResult, InstanceInfo, SearchResult, SourceInfo, SourceStatus
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
        # Serialises writes within this process. The manifest is a read-modify-write
        # over a plain file — two concurrent index runs would lose one another's
        # entries, and SQLite/WAL does not cover it because it is not in the store.
        # Reentrant: index() holds it while calling helpers that take it too.
        # NOTE: process-local. Two processes on one store still need external
        # coordination — see README.
        self._write_lock = threading.RLock()

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
            # Absorbing this used to silently re-embed the whole source and leave
            # scan/status permanently wrong. Fail loudly; deleting the file is a
            # deliberate act, not something the library should do on your behalf.
            raise ManifestCorrupt(
                f"Manifest at {p} is not valid JSON ({e}). The index itself is intact — "
                f"delete the file to rebuild it from the next full index run."
            ) from e

    def _save_manifest(self, manifest: dict) -> None:
        """Write the manifest atomically — temp file in the same directory, then
        os.replace. A plain write_text truncates first, so a crash mid-write leaves
        a zero-byte file where a valid one used to be."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        target = self._manifest_path()
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)          # atomic within one filesystem

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
            logger.warning("embedding function changed for %r — clearing old index", collection_name)
            client.delete_collection(collection_name)
            return client.create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
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
        """Index a DataSource into its own ChromaDB collection.

        Returns an IndexResult describing what happened — including `aborted=True`
        when the mass-change guard stopped the run, and `errors` listing any files
        that could not be parsed. Nothing is printed; pass `on_progress` for a live
        feed and `on_confirm` to be asked about a mass change.

        `limit` caps the number of files (first N, stable order). `pause_ms` and
        `pause_every` throttle CPU duty between embedded files.

        The manifest is written even if the run raises partway, recording only the
        files that actually completed — so an interrupted run resumes rather than
        starting over.
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

        client = self._client()
        if force:
            try:
                client.delete_collection(source.source_id)
                self._emit(on_progress, f"Cleared existing index for {source.source_id!r}.")
            except Exception as e:
                # Nothing to clear is the normal case on a first run; anything else
                # is worth a log line rather than a silent pass.
                logger.debug("force delete_collection(%s): %s", source.source_id, e)

        col = self._get_collection(client, source.source_id)

        # Snapshot what's already indexed, keyed by each file's relative path.
        snap = col.get(include=["metadatas"])
        ids_by_file: dict[str, list[str]] = {}
        for cid, meta in zip(snap["ids"], snap["metadatas"]):
            rp = meta.get("rel_path") or meta.get("file") or cid.split("::")[0]
            ids_by_file.setdefault(rp, []).append(cid)

        # Pre-scan: hash every file once and decide the real work up front.
        prev_manifest: dict = {} if force else self._load_manifest().get(source.source_id, {})
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

        # Manifest entry built as work completes, so an interrupted run records only
        # what really finished. A full run starts empty (deleted files drop out); a
        # --limit run starts from the previous entry, since it only sees a subset.
        done: dict[str, str] = {} if limit is None else dict(prev_manifest)
        for _, rp, h in entries:
            if prev_manifest.get(rp) == h:
                done[rp] = h                      # unchanged: already correct

        since_pause = 0
        try:
            for processed, (f, rel_path, file_hash) in enumerate(to_process, 1):
                was_indexed = rel_path in ids_by_file
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

                if not chunks:
                    result.empty += 1
                    if was_indexed:
                        col.delete(ids=ids_by_file[rel_path])
                    done[rel_path] = file_hash
                    continue

                verb = "re-embedding" if was_indexed else "embedding"
                self._emit(on_progress,
                           f"  [{processed}/{work_total}] {verb} {rel_path}  ({len(chunks)} chunks)")

                if was_indexed:
                    col.delete(ids=ids_by_file[rel_path])

                new_ids, new_docs, new_metas = [], [], []
                for chunk in chunks:
                    new_ids.append(f"{rel_path}::{chunk.id_suffix}")
                    new_docs.append(chunk.text)
                    new_metas.append({**chunk.metadata, "rel_path": rel_path,
                                      "content_hash": file_hash})
                col.add(documents=new_docs, metadatas=new_metas, ids=new_ids)
                done[rel_path] = file_hash
                result.updated += 1 if was_indexed else 0
                result.added += 0 if was_indexed else 1

                if pause_ms > 0 and pause_every > 0:
                    since_pause += 1
                    if since_pause >= pause_every:
                        time.sleep(pause_ms / 1000.0)
                        since_pause = 0

            # Prune files that no longer exist on disk. Full runs only — a --limit run
            # sees a subset, so absent files there are not actually deleted.
            if limit is None and deleted_files:
                stale_ids = [cid for rp in deleted_files if rp in ids_by_file
                             for cid in ids_by_file[rp]]
                if stale_ids:
                    col.delete(ids=stale_ids)
                result.pruned = len(deleted_files)
        finally:
            # Always persist what completed, including on an exception. Without this a
            # crash at file 400 of 450 throws away 400 files of work.
            manifest = self._load_manifest()
            manifest[source.source_id] = done
            self._save_manifest(manifest)

        result.total_chunks = col.count()
        self._emit(on_progress, (
            f"\nDone [{source.source_id}].\n"
            f"  files:  new={result.added} updated={result.updated} "
            f"unchanged={result.unchanged} empty={result.empty} pruned={result.pruned}"
            f"  ({result.embedded} embedded this run)\n"
            f"  chunks: {result.total_chunks} total in index"
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
        manifest = self._load_manifest().get(source.source_id, {})
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
        Callers must serialize this (one writer) — ChromaDB is not multi-writer safe.
        """
        with self._write_lock:
            return self._reindex_paths_locked(source, paths, chunk_size, overlap, min_chunk)

    def _reindex_paths_locked(self, source, paths, chunk_size, overlap, min_chunk) -> dict:
        rels: dict[str, Path] = {}
        for p in paths:
            p = Path(p)
            rels[self._rel_path(source, p)] = p
        counts = {"embedded": 0, "empty": 0, "pruned": 0, "unchanged": 0}
        if not rels:
            return counts

        client = self._client()
        col = self._get_collection(client, source.source_id)
        chunker = source.make_chunker(chunk_size, overlap, min_chunk)

        # Existing chunk ids for just these files. rel_path metadata exists for anything
        # (re)indexed under the manifest scheme, which the watcher's startup reconcile
        # guarantees before per-file events are handled.
        ids_by_file: dict[str, list[str]] = {}
        got = col.get(where={"rel_path": {"$in": list(rels)}}, include=["metadatas"])
        for cid, meta in zip(got["ids"], got["metadatas"]):
            rp = meta.get("rel_path") or cid.split("::")[0]
            ids_by_file.setdefault(rp, []).append(cid)

        manifest = self._load_manifest()
        src_manifest = manifest.setdefault(source.source_id, {})
        for rp, p in rels.items():
            old_ids = ids_by_file.get(rp, [])
            if not p.exists():                       # deleted → prune + forget
                if old_ids:
                    col.delete(ids=old_ids)
                src_manifest.pop(rp, None)
                counts["pruned"] += 1
                logger.info("reindex_paths: pruned %s/%s", source.source_id, rp)
                continue

            file_hash = _file_hash(p)
            # Unchanged content already in the index: nothing to do. Guards against
            # mtime-only touches and spurious watcher events; a re-embed of identical
            # text just bloats the HNSW index (deleted vectors are never reclaimed).
            if old_ids and src_manifest.get(rp) == file_hash:
                counts["unchanged"] += 1
                logger.debug("reindex_paths: unchanged %s/%s", source.source_id, rp)
                continue
            doc = source.parse_file(p)
            chunks = chunker.chunk(doc) if doc is not None else []
            if old_ids:                              # replace: drop old chunks first
                col.delete(ids=old_ids)
            src_manifest[rp] = file_hash             # track even if it yields no chunks
            if not chunks:
                counts["empty"] += 1
                continue

            new_ids, new_docs, new_metas = [], [], []
            for chunk in chunks:
                new_ids.append(f"{rp}::{chunk.id_suffix}")
                new_docs.append(chunk.text)
                new_metas.append({**chunk.metadata, "rel_path": rp, "content_hash": file_hash})
            col.add(documents=new_docs, metadatas=new_metas, ids=new_ids)
            counts["embedded"] += 1
            logger.info("reindex_paths: embedded %s/%s (%d chunks)", source.source_id, rp, len(chunks))

        self._save_manifest(manifest)
        return counts

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
        if not self.chroma_dir.exists():
            raise IndexNotFound(
                f"No index at {self.chroma_dir}. Run: python -m basic_kb index"
            )
        client = self._client()
        return self._run_query_group(
            client, sources, queries, n, content_type_filter, rerank_candidates,
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
        if not self.chroma_dir.exists():
            raise IndexNotFound(
                f"No index at {self.chroma_dir}. Run: python -m basic_kb index"
            )
        client = self._client()
        out: list[tuple[str, list[SearchResult]]] = []
        for q in queries:
            hits = self._run_query_group(
                client, sources, [q], n, content_type_filter, rerank_candidates,
                cand_multiplier, cand_min, cand_max, strict_rerank, timing, timing_label=q,
            )
            out.append((q, hits))
        return out

    def _run_query_group(
        self, client, sources: list[DataSourceBase], queries: list[str], n: int,
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
        missing: list[str] = []      # sources with no collection yet
        resolved = 0                 # sources that actually got queried

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
            except Exception as e:
                # One un-indexed source among several is a normal, recoverable state.
                # Every source missing is not — that is caught after the loop.
                logger.warning("search: no index for source=%s (%s)", source.source_id, e)
                missing.append(source.source_id)
                continue
            resolved += 1

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
                    # Swallowing this returned an empty list indistinguishable from
                    # "nothing matched" — a broken store looked like a healthy one.
                    raise QueryFailed(
                        f"Query failed on source {source.source_id!r}: {e}"
                    ) from e

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
        client = self._client() if self.chroma_dir.exists() else None
        out: list[SourceInfo] = []

        for source in sources:
            chunks, indexed = 0, False
            if client is not None:
                try:
                    col = client.get_collection(
                        name=source.source_id,
                        embedding_function=self.embedder.as_chroma_ef(),
                    )
                    chunks = col.count()
                    indexed = chunks > 0
                except Exception as e:
                    logger.info("info: no collection for source=%s (%s)", source.source_id, e)

            out.append(SourceInfo(
                source_id=source.source_id,
                label=source.label,
                description=source.description,
                type=type(source).type_name or type(source).__name__,
                chunker=source.chunker_name,
                files=len(source.get_files()),
                chunks=chunks,
                indexed=indexed,
            ))

        return InstanceInfo(
            name=name,
            model_id=self.embedder.model_id,
            store_dir=str(self.chroma_dir),
            sources=out,
        )

    def status(self, sources: list[DataSourceBase]) -> list[SourceStatus]:
        """Index stats per source, as data. Nothing is printed — the CLI formats it.

        Raises IndexNotFound if the store does not exist at all; a source that is
        merely un-indexed comes back with `indexed=False` rather than an exception,
        because that is a normal state in a multi-source instance.
        """
        if not self.chroma_dir.exists():
            raise IndexNotFound(
                f"No index at {self.chroma_dir}. Run: python -m basic_kb index"
            )

        client = self._client()
        out: list[SourceStatus] = []

        for source in sources:
            st = SourceStatus(
                source_id=source.source_id,
                label=source.label,
                directory=str(source.directory),
                store_dir=str(self.chroma_dir),
                model_id=self.embedder.model_id,
                directory_exists=source.directory.exists(),
                indexed=False,
            )

            try:
                col = client.get_collection(
                    name=source.source_id,
                    embedding_function=self.embedder.as_chroma_ef(),
                )
            except Exception as e:
                logger.info("status: no collection for source=%s (%s)", source.source_id, e)
                out.append(st)
                continue

            st.indexed = True
            st.chunks = col.count()

            res = self.scan(source)
            st.files_on_disk = res.files_on_disk
            st.tracked = res.tracked
            st.new, st.updated, st.deleted = res.new, res.updated, res.deleted

            if st.chunks == 0:
                out.append(st)
                continue

            all_metas: list[dict] = col.get(include=["metadatas"])["metadatas"]
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
