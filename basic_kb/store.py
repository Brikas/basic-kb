"""SqliteVecStore — vectors, chunk text, metadata and the file manifest in one SQLite file.

Why sqlite-vec and exact search instead of an HNSW store: see docs/adr/0001. In short,
the corpora this engine serves are small (thousands to low hundreds of thousands of
chunks) and update-heavy; a brute-force cosine scan is milliseconds at that size, a
delete is a real delete, and nothing has to live in RAM.

Layout (one database per instance, `<store_dir>/kb.sqlite3`):

    meta   (key, value)                       model id, vector dimension, schema version,
                                               rows deleted since the last VACUUM
    files  (source, rel_path, hash)           every file seen at index time -> content hash
                                               (the old manifest.json); powers scan/stale
    chunks (id, source, rel_path, chunk_id, doc, meta_json, content_type, date)
    vec    vec0(source partition key, content_type, embedding)  rowid == chunks.id

Every public method opens its own connection and closes it, so the store is safe to
call from any thread; writers still have to be serialised by the caller (SQLite has
one writer at a time — the KnowledgeBase holds that lock).

Auto-vacuum: SQLite does not shrink a file when rows are deleted, it only marks pages
free. The store counts rows deleted since the last VACUUM and, once that count is at
least `vacuum_min_deleted` AND at least `vacuum_deleted_fraction` of the live rows,
runs VACUUM at the end of the write that crossed the line. This lives here, not in the
CLI, so the watcher and library callers get it too.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("basic_kb")

SCHEMA_VERSION = 1
DB_FILENAME = "kb.sqlite3"

# TEMPORARY (added 2026-08-29, remove once every instance has been re-indexed on
# sqlite-vec): files that identify a pre-ADR-0001 ChromaDB store in the same directory.
LEGACY_CHROMA_MARKERS = ("chroma.sqlite3", "manifest.json")


from .errors import StoreError  # noqa: E402  (re-exported for callers that import it from here)


@dataclass
class VacuumPolicy:
    """When to run VACUUM after a write. Mirrors the `vacuum:` config block."""
    enabled: bool = True
    deleted_fraction: float = 0.2   # deleted-since-vacuum / live rows that triggers it
    min_deleted: int = 1000         # ...but never for fewer deleted rows than this


@dataclass
class ChunkRow:
    """One chunk as the store returns it from a KNN query."""
    chunk_id: str        # "<rel_path>::<suffix>", unique within a source
    doc: str
    metadata: dict
    distance: float      # cosine distance; score = 1 - distance


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _load_extension(con: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
    except ImportError as e:
        raise StoreError("sqlite-vec is not installed: pip install sqlite-vec") from e
    enable = getattr(con, "enable_load_extension", None)
    if enable is None:
        # Some Python builds (historically python.org macOS) ship sqlite3 without
        # loadable-extension support. Nothing to work around at runtime; say so.
        raise StoreError(
            "this Python's sqlite3 module cannot load extensions "
            "(no Connection.enable_load_extension). Use a Python built with "
            "extension support (Homebrew, uv-managed, python.org >= 3.13, conda)."
        )
    enable(True)
    sqlite_vec.load(con)
    enable(False)


class SqliteVecStore:
    def __init__(self, store_dir: Path, vacuum: Optional[VacuumPolicy] = None) -> None:
        self.store_dir = Path(store_dir)
        self.path = self.store_dir / DB_FILENAME
        self.vacuum_policy = vacuum or VacuumPolicy()

    # --- connections -------------------------------------------------------------
    def exists(self) -> bool:
        return self.path.exists()

    def _connect(self, create: bool = False) -> sqlite3.Connection:
        if not create and not self.path.exists():
            raise FileNotFoundError(self.path)
        if create:
            self.store_dir.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=30)
        _load_extension(con)
        con.execute("PRAGMA journal_mode=WAL")    # readers never block the one writer
        con.execute("PRAGMA synchronous=NORMAL")
        if create:
            self._ensure_schema(con)
        return con

    def _ensure_schema(self, con: sqlite3.Connection) -> None:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files (
                source TEXT NOT NULL, rel_path TEXT NOT NULL, hash TEXT NOT NULL,
                PRIMARY KEY (source, rel_path));
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL, rel_path TEXT NOT NULL, chunk_id TEXT NOT NULL,
                doc TEXT NOT NULL, meta_json TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '',
                UNIQUE (source, chunk_id));
            CREATE INDEX IF NOT EXISTS chunks_source_rel ON chunks (source, rel_path);
            """
        )
        con.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        con.execute("INSERT OR IGNORE INTO meta VALUES ('deleted_since_vacuum', '0')")
        con.commit()

    @staticmethod
    def _meta_get(con: sqlite3.Connection, key: str) -> Optional[str]:
        row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _meta_set(con: sqlite3.Connection, key: str, value: str) -> None:
        con.execute("INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value))

    def model_info(self) -> Optional[tuple[str, int]]:
        """(model_id, dim) the store was built with, or None if nothing embedded yet."""
        if not self.exists():
            return None
        with self._connect() as con:
            m, d = self._meta_get(con, "model_id"), self._meta_get(con, "dim")
        return (m, int(d)) if m and d else None

    def check_model(self, model_id: str) -> None:
        """Refuse to read or write with a different embedding model than the store holds.

        Vectors from two models share no geometry, so a search with the wrong model
        would return confident nonsense rather than fail. Raise instead.
        """
        info = self.model_info()
        if info and info[0] != model_id:
            raise StoreError(
                f"store at {self.path} was built with embedding model {info[0]!r}; "
                f"this run uses {model_id!r}. Search results would be meaningless. Either use "
                f"the original model, or rebuild every source: `basic_kb index --force --source all`.")

    def _reset_vectors(self, con: sqlite3.Connection) -> None:
        """Drop the vector table and forget model/dim — only valid when no chunks remain."""
        con.execute("DROP TABLE IF EXISTS vec")
        con.execute("DELETE FROM meta WHERE key IN ('dim', 'model_id')")

    def _ensure_vec_table(self, con: sqlite3.Connection, dim: int, model_id: str) -> None:
        """Create the vector table on first write; refuse silently mixing models/dims.

        The dimension is fixed at creation. Switching models therefore requires the
        store to be empty first — `index --force --source all` clears every source,
        which resets the table; a force on one source alone is refused if others
        still hold vectors from the old model.
        """
        have_dim = self._meta_get(con, "dim")
        have_model = self._meta_get(con, "model_id")
        if have_dim is not None and (int(have_dim) != dim or have_model != model_id):
            if con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0:
                self._reset_vectors(con)
                have_dim = None
        if have_dim is None:
            con.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0("
                f"source TEXT PARTITION KEY, content_type TEXT, "
                f"embedding FLOAT[{dim}] distance_metric=cosine)")
            self._meta_set(con, "dim", str(dim))
            self._meta_set(con, "model_id", model_id)
            con.commit()
            return
        if int(have_dim) != dim or have_model != model_id:
            raise StoreError(
                f"store at {self.path} was built with model {have_model!r} (dim {have_dim}); "
                f"now embedding with {model_id!r} (dim {dim}) while other sources still hold the old "
                f"vectors. Run `basic_kb index --force --source all` to rebuild everything, or keep "
                f"the original model.")

    # --- manifest (files table) ----------------------------------------------------
    def manifest(self, source: str) -> dict[str, str]:
        """{rel_path: content hash} for one source; {} if nothing indexed yet."""
        if not self.exists():
            return {}
        with self._connect() as con:
            return dict(con.execute("SELECT rel_path, hash FROM files WHERE source = ?", (source,)).fetchall())

    def source_indexed(self, source: str) -> bool:
        if not self.exists():
            return False
        with self._connect() as con:
            return con.execute("SELECT 1 FROM files WHERE source = ? LIMIT 1", (source,)).fetchone() is not None

    def count(self, source: str) -> int:
        if not self.exists():
            return 0
        with self._connect() as con:
            return con.execute("SELECT count(*) FROM chunks WHERE source = ?", (source,)).fetchone()[0]

    def chars(self, source: str) -> int:
        """Total characters of chunk text stored for a source (≈ tokens × 4)."""
        if not self.exists():
            return 0
        with self._connect() as con:
            return con.execute("SELECT coalesce(sum(length(doc)), 0) FROM chunks WHERE source = ?", (source,)).fetchone()[0]

    def chunk_ids_by_file(self, source: str, rel_paths: Optional[Iterable[str]] = None) -> dict[str, list[int]]:
        """{rel_path: [chunk row ids]} for a source, optionally only for some files."""
        if not self.exists():
            return {}
        out: dict[str, list[int]] = {}
        with self._connect() as con:
            if rel_paths is None:
                rows = con.execute("SELECT rel_path, id FROM chunks WHERE source = ?", (source,)).fetchall()
            else:
                rels = list(rel_paths)
                rows = []
                for i in range(0, len(rels), 500):          # stay under SQLite's variable limit
                    part = rels[i:i + 500]
                    marks = ",".join("?" * len(part))
                    rows += con.execute(
                        f"SELECT rel_path, id FROM chunks WHERE source = ? AND rel_path IN ({marks})",
                        (source, *part)).fetchall()
        for rp, cid in rows:
            out.setdefault(rp, []).append(cid)
        return out

    def all_metadata(self, source: str) -> list[dict]:
        if not self.exists():
            return []
        with self._connect() as con:
            return [json.loads(m) for (m,) in
                    con.execute("SELECT meta_json FROM chunks WHERE source = ?", (source,)).fetchall()]

    # --- writes (caller serialises) --------------------------------------------------
    def clear_source(self, source: str) -> int:
        """Drop everything indexed for one source. Returns rows deleted."""
        if not self.exists():
            return 0
        with self._connect() as con:
            n = con.execute("SELECT count(*) FROM chunks WHERE source = ?", (source,)).fetchone()[0]
            if self._meta_get(con, "dim") is not None:
                con.execute("DELETE FROM vec WHERE source = ?", (source,))
            con.execute("DELETE FROM chunks WHERE source = ?", (source,))
            con.execute("DELETE FROM files WHERE source = ?", (source,))
            self._bump_deleted(con, n)
            if con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0:
                self._reset_vectors(con)          # empty store: a new model may follow
            con.commit()
        self._maybe_vacuum()
        return n

    def sync_file(
        self, source: str, rel_path: str, file_hash: str,
        chunk_ids: list[str], docs: list[str], metadatas: list[dict],
        embed, model_id: str,
    ) -> tuple[int, int]:
        """Bring one file's rows in line with its current chunks, embedding only what is new.

        Chunk ids are content-derived (see KnowledgeBase._chunk_ids), so a chunk whose text
        is unchanged keeps its row and vector; only ids absent from the store are embedded
        (`embed(texts) -> vectors` is called once, for those). Rows whose id disappeared are
        deleted; kept rows get their metadata refreshed (position, file hash). Appending to a
        file therefore costs one or two embeddings, not the whole file. Returns
        (chunks_embedded, chunks_reused). An empty chunk list drops every row for the file.
        """
        with self._connect(create=True) as con:
            have = {cid: (rid, ct) for rid, cid, ct in con.execute(
                "SELECT id, chunk_id, content_type FROM chunks WHERE source = ? AND rel_path = ?",
                (source, rel_path)).fetchall()}
            wanted = set(chunk_ids)
            gone = [rid for cid, (rid, _) in have.items() if cid not in wanted]
            if gone:
                self._delete_rows(con, gone)

            new_idx = [i for i, cid in enumerate(chunk_ids) if cid not in have]
            embeddings = embed([docs[i] for i in new_idx]) if new_idx else []
            if embeddings:
                self._ensure_vec_table(con, len(embeddings[0]), model_id)

            for i, cid in enumerate(chunk_ids):
                meta = metadatas[i]
                ct = str(meta.get("content_type") or "")
                if cid in have:
                    rid, old_ct = have[cid]
                    con.execute("UPDATE chunks SET meta_json = ?, content_type = ?, date = ? WHERE id = ?",
                                (json.dumps(meta, ensure_ascii=False), ct, str(meta.get("date") or ""), rid))
                    if old_ct != ct:
                        # vec0 metadata columns are filterable at query time, so they must be
                        # right; re-insert the row with its existing vector instead of re-embedding.
                        (blob,) = con.execute("SELECT embedding FROM vec WHERE rowid = ?", (rid,)).fetchone()
                        con.execute("DELETE FROM vec WHERE rowid = ?", (rid,))
                        con.execute("INSERT INTO vec (rowid, source, content_type, embedding) VALUES (?, ?, ?, ?)",
                                    (rid, source, ct, blob))
                    continue
                cur = con.execute(
                    "INSERT INTO chunks (source, rel_path, chunk_id, doc, meta_json, content_type, date) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (source, rel_path, cid, docs[i], json.dumps(meta, ensure_ascii=False), ct,
                     str(meta.get("date") or "")))
                con.execute("INSERT INTO vec (rowid, source, content_type, embedding) VALUES (?, ?, ?, ?)",
                            (cur.lastrowid, source, ct, _pack(embeddings[new_idx.index(i)])))
            con.execute(
                "INSERT INTO files VALUES (?, ?, ?) ON CONFLICT(source, rel_path) DO UPDATE SET hash = excluded.hash",
                (source, rel_path, file_hash))
            con.commit()
        self._maybe_vacuum()
        return len(new_idx), len(chunk_ids) - len(new_idx)

    def remove_files(self, source: str, rel_paths: Iterable[str]) -> int:
        """Prune files that vanished from disk: chunks, vectors and manifest entry. Returns files removed."""
        rels = list(rel_paths)
        if not rels or not self.exists():
            return 0
        with self._connect() as con:
            for rp in rels:
                ids = [cid for (cid,) in con.execute(
                    "SELECT id FROM chunks WHERE source = ? AND rel_path = ?", (source, rp)).fetchall()]
                if ids:
                    self._delete_rows(con, ids)
                con.execute("DELETE FROM files WHERE source = ? AND rel_path = ?", (source, rp))
            con.commit()
        self._maybe_vacuum()
        return len(rels)

    def _delete_rows(self, con: sqlite3.Connection, ids: list[int]) -> None:
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            if self._meta_get(con, "dim") is not None:
                con.execute(f"DELETE FROM vec WHERE rowid IN ({marks})", part)
            con.execute(f"DELETE FROM chunks WHERE id IN ({marks})", part)
        self._bump_deleted(con, len(ids))

    def _bump_deleted(self, con: sqlite3.Connection, n: int) -> None:
        if n <= 0:
            return
        cur = int(self._meta_get(con, "deleted_since_vacuum") or 0)
        self._meta_set(con, "deleted_since_vacuum", str(cur + n))

    # --- vacuum ----------------------------------------------------------------------
    def vacuum_stats(self) -> dict:
        """{live, deleted_since_vacuum, fraction, would_vacuum} — for status and tests."""
        if not self.exists():
            return {"live": 0, "deleted_since_vacuum": 0, "fraction": 0.0, "would_vacuum": False}
        with self._connect() as con:
            live = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
            deleted = int(self._meta_get(con, "deleted_since_vacuum") or 0)
        frac = deleted / live if live else (1.0 if deleted else 0.0)
        p = self.vacuum_policy
        return {"live": live, "deleted_since_vacuum": deleted, "fraction": round(frac, 4),
                "would_vacuum": bool(p.enabled and deleted >= p.min_deleted and frac >= p.deleted_fraction)}

    def _maybe_vacuum(self) -> None:
        st = self.vacuum_stats()
        if st["would_vacuum"]:
            self.vacuum(reason=f"deleted={st['deleted_since_vacuum']} live={st['live']} "
                               f"fraction={st['fraction']:.2f} >= {self.vacuum_policy.deleted_fraction}")

    def vacuum(self, reason: str = "requested") -> bool:
        """Run VACUUM now and reset the deleted counter. Returns False if the database was busy.

        VACUUM needs the database to itself for a moment; a concurrent reader makes
        SQLite raise `database is locked`. That is not an error worth failing a write
        over — log it and let the next write try again.
        """
        if not self.exists():
            return False
        t = time.perf_counter()
        before = self.path.stat().st_size
        try:
            with self._connect() as con:
                con.execute("VACUUM")
                self._meta_set(con, "deleted_since_vacuum", "0")
                con.commit()
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError as e:
            logger.warning("vacuum skipped (%s): %s", reason, e)
            return False
        after = self.path.stat().st_size
        logger.info("vacuum done (%s): %s -> %s bytes in %.0f ms", reason, f"{before:,}", f"{after:,}",
                    (time.perf_counter() - t) * 1000)
        return True

    # --- search ----------------------------------------------------------------------
    def knn(self, source: str, query: list[float], k: int,
            content_type: Optional[str] = None) -> list[ChunkRow]:
        """Exact top-k by cosine distance within one source, optionally one content_type."""
        if not self.exists():
            return []
        sql = ("SELECT v.rowid, v.distance, c.chunk_id, c.doc, c.meta_json "
               "FROM vec v JOIN chunks c ON c.id = v.rowid "
               "WHERE v.embedding MATCH ? AND k = ? AND v.source = ?")
        params: list = [_pack(query), int(k), source]
        if content_type:
            sql += " AND v.content_type = ?"
            params.append(content_type)
        with self._connect() as con:
            if self._meta_get(con, "dim") is None:
                return []
            rows = con.execute(sql, params).fetchall()
        return [ChunkRow(chunk_id=cid, doc=doc, metadata=json.loads(mj), distance=float(dist))
                for _, dist, cid, doc, mj in rows]

    # --- TEMPORARY: legacy Chroma detection ----------------------------------------------
    def legacy_chroma_leftovers(self) -> list[Path]:
        """Files/dirs from the pre-ADR-0001 ChromaDB store still sitting in store_dir.

        TEMPORARY (2026-08-29): exists only so every machine/instance gets told to rebuild
        once. Delete this method, LEGACY_CHROMA_MARKERS and the CLI notice once all
        instances have been migrated.
        """
        # `.chroma` was the historical default store_dir; an instance that renamed its
        # store_dir to `.basic-kb` still has the old directory sitting beside it.
        candidates = {self.store_dir, self.store_dir.parent / ".chroma"}
        found: list[Path] = []
        for d in candidates:
            if not d.exists():
                continue
            found += [d / m for m in LEGACY_CHROMA_MARKERS if (d / m).exists()]
            if (d / "chroma.sqlite3").exists():
                # Chroma keeps one UUID-named directory per collection segment.
                found += [p for p in d.iterdir()
                          if p.is_dir() and len(p.name) == 36 and p.name.count("-") == 4]
        return sorted(set(found))
