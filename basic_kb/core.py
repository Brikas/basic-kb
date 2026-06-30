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
    ) -> None:
        """Index all files from a DataSource into its own ChromaDB collection.

        `limit` caps the number of files (first N, stable order) — handy for test
        runs before committing to a long full embed. `pause_ms`/`pause_every` throttle
        CPU duty by sleeping between batches of embedded files.
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
        # rel_path is the stable per-file identity; content_hash detects edits.
        snap = col.get(include=["metadatas"])
        stored_hash: dict[str, Optional[str]] = {}   # rel_path -> indexed content hash
        ids_by_file: dict[str, list[str]] = {}       # rel_path -> its chunk ids
        for cid, meta in zip(snap["ids"], snap["metadatas"]):
            # Fall back to legacy 'file' key for chunks indexed before rel_path existed.
            rp = meta.get("rel_path") or meta.get("file") or cid.split("::")[0]
            stored_hash.setdefault(rp, meta.get("content_hash"))
            ids_by_file.setdefault(rp, []).append(cid)

        chunker = source.make_chunker(chunk_size, overlap, min_chunk)
        print(
            f"Indexing {len(files)} {source.label} files  "
            f"(model={self.embedder.model_id}, chunker={chunker.name}, "
            f"chunk={chunk_size}, overlap={overlap})"
        )
        logger.info("index start: source=%s files=%d model=%s force=%s limit=%s",
                    source.source_id, len(files), self.embedder.model_id, force, limit)

        added = updated = unchanged = empty = pruned = 0
        total_files = len(files)
        seen: dict[str, str] = {}   # rel_path -> hash, for the manifest (all files, incl. empty)
        since_pause = 0             # embedded files since the last throttle pause

        for i, f in enumerate(files, 1):
            rel_path = self._rel_path(source, f)
            file_hash = _file_hash(f)
            seen[rel_path] = file_hash

            # Unchanged: same content already indexed → skip without parsing/embedding.
            if stored_hash.get(rel_path) == file_hash and rel_path in ids_by_file:
                unchanged += 1
                continue

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
            print(f"  [{i}/{total_files}] {verb} {rel_path}  ({len(chunks)} chunks)", flush=True)

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
        if limit is None:
            stale_ids = [cid for rp, ids in ids_by_file.items() if rp not in seen for cid in ids]
            if stale_ids:
                col.delete(ids=stale_ids)
                pruned = sum(1 for rp in ids_by_file if rp not in seen)

        # Persist the manifest. Full run replaces this source's entry (drops deleted
        # files); a --limit run merges, since it only saw a subset.
        manifest = self._load_manifest()
        if limit is None:
            manifest[source.source_id] = seen
        else:
            manifest[source.source_id] = {**manifest.get(source.source_id, {}), **seen}
        self._save_manifest(manifest)

        total = col.count()
        embedded = added + updated + unchanged   # files that contributed chunks to the index
        print(
            f"\nDone [{source.source_id}].\n"
            f"  files:  new={added} updated={updated} unchanged={unchanged} "
            f"empty={empty} pruned={pruned}  ({embedded} of {total_files} indexed)\n"
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
            indexed_files: set[str] = {m["file"] for m in all_metas}
            files_on_disk = len(source.get_files())
            unindexed = files_on_disk - len(indexed_files)

            print(f"Chunks  : {total_chunks:,}")
            print(f"Docs    : {len(indexed_files):,} indexed / {files_on_disk} on disk  ({unindexed} not indexed)")

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
