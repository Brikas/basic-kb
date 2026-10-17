"""SqliteVecStore — vectors, chunk text, metadata and the file manifest in one SQLite file.

Why sqlite-vec and exact search instead of an HNSW store: see docs/adr/0001. In short,
the corpora this engine serves are small (thousands to low hundreds of thousands of
chunks) and update-heavy; a brute-force cosine scan is milliseconds at that size, a
delete is a real delete, and nothing has to live in RAM.

Layout (one database per instance, `<store_dir>/kb.sqlite3`):

    meta   (key, value)                       model id, vector dimension, schema version,
                                               rows deleted since the last VACUUM, and
                                               `indexed_at:<source>` per-source run times
    files  (source, rel_path, hash)           every file seen at index time -> content hash
                                               (the old manifest.json); powers scan/stale
    chunks (id, source, rel_path, chunk_id, doc, meta_json, content_type, date)
    vec    vec0(source partition key, content_type, embedding)  rowid == chunks.id
    chunks_fts  fts5(source, doc), external content over `chunks`, rowid == chunks.id;
                kept in step by triggers on `chunks` (keyword side of hybrid search)

Never write `chunks` with INSERT OR REPLACE: the implicit delete skips the FTS trigger and
leaves the keyword index corrupt. Plain INSERT, UPDATE and DELETE keep it consistent.

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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .lexical import FTS_TOKENIZER, source_filter

logger = logging.getLogger("basic_kb")

SCHEMA_VERSION = 2   # 2: chunks_fts keyword index; a version-1 store builds it on first write
DB_FILENAME = "kb.sqlite3"

_FTS_SCHEMA = f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        source, doc, content='chunks', content_rowid='id', tokenize='{FTS_TOKENIZER}');
    CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, source, doc) VALUES (new.id, new.source, new.doc);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, source, doc) VALUES ('delete', old.id, old.source, old.doc);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE OF source, doc ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, source, doc) VALUES ('delete', old.id, old.source, old.doc);
        INSERT INTO chunks_fts(rowid, source, doc) VALUES (new.id, new.source, new.doc);
    END;
"""

from .errors import StoreError  # noqa: E402  (re-exported for callers that import it from here)


@dataclass
class VacuumPolicy:
    """When to run VACUUM after a write. Mirrors the `vacuum:` config block."""
    enabled: bool = True
    deleted_fraction: float = 0.2   # deleted-since-vacuum / live rows that triggers it
    min_deleted: int = 1000         # ...but never for fewer deleted rows than this


@dataclass
class ChunkRow:
    """One chunk as the store returns it from a vector or keyword query."""
    chunk_id: str        # "<rel_path>::<suffix>", unique within a source
    doc: str
    metadata: dict
    distance: Optional[float] = None   # cosine distance (score = 1 - distance); None from a keyword query
    rowid: int = 0                     # chunks.id, shared by vec and chunks_fts
    source: str = ""
    bm25: Optional[float] = None       # FTS5 bm25(), lower is better; keyword queries only


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

    @contextmanager
    def _connect(self, create: bool = False) -> Iterator[sqlite3.Connection]:
        """One connection for one `with` block: committed on success, rolled back on
        error, and closed on exit.

        sqlite3.Connection's own context manager commits but leaves the connection open
        until garbage collection. An open connection keeps the -wal and -shm files alive,
        so a long-lived process (a server) whose store file is replaced by a fresh build
        reads the new file through the old WAL and fails with `disk I/O error`.
        """
        if not create and not self.path.exists():
            raise FileNotFoundError(self.path)
        if create:
            self.store_dir.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=30)
        try:
            _load_extension(con)
            con.execute("PRAGMA journal_mode=WAL")    # readers never block the one writer
            con.execute("PRAGMA synchronous=NORMAL")
            if create:
                self._ensure_schema(con)
            with con:
                yield con
        finally:
            con.close()

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
        self._ensure_fts(con)

    def _ensure_fts(self, con: sqlite3.Connection) -> None:
        """Create the keyword index and its triggers, filling it from existing chunks once.

        A store written before schema 2 has chunks but no index; `rebuild` reads every
        chunk's text back from `chunks`, so nothing is re-embedded. A Python whose SQLite
        lacks FTS5 is refused with the fix rather than left without keyword search.
        """
        if self._has_table(con, "chunks_fts"):
            return
        t = time.perf_counter()
        try:
            con.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError as e:
            if "fts5" in str(e).lower():
                raise StoreError(
                    f"this Python's SQLite ({sqlite3.sqlite_version}) has no FTS5, which keyword and "
                    f"hybrid search need. python.org, Homebrew, Debian and the official Docker "
                    f"images all ship it; use one of those.") from e
            raise
        n = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if n:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
        self._meta_set(con, "schema_version", str(SCHEMA_VERSION))
        con.commit()
        logger.info("store: keyword index built over %d chunks in %.0f ms", n, (time.perf_counter() - t) * 1000)

    @staticmethod
    def _has_table(con: sqlite3.Connection, name: str) -> bool:
        return con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None

    def keyword_index_ready(self) -> bool:
        """True when the keyword index exists. A store from before schema 2 gets it on its
        next write, or from `ensure_schema()`; until then keyword search would find nothing."""
        if getattr(self, "_fts_ready", False):
            return True                      # the table is never dropped once it exists
        if not self.exists():
            return False
        with self._connect() as con:
            self._fts_ready = self._has_table(con, "chunks_fts")
        return self._fts_ready

    def ensure_schema(self) -> None:
        """Bring the store to the current schema now (a write: the caller holds the locks)."""
        with self._connect(create=True):
            pass

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

    def set_indexed_at(self, source: str, ts: Optional[float] = None) -> None:
        """Stamp when this source finished an index run (epoch seconds, in `meta`).

        Called by the writer after the manifest is up to date, so a reader can say how
        old the index is without inspecting file mtimes.
        """
        with self._connect() as con:
            self._meta_set(con, f"indexed_at:{source}", repr(ts if ts is not None else time.time()))
            con.commit()

    def indexed_at(self, source: str) -> Optional[float]:
        """Epoch seconds of this source's last index run, or None if never stamped
        (a store written before stamping existed — it re-stamps on the next index)."""
        if not self.exists():
            return None
        with self._connect() as con:
            raw = self._meta_get(con, f"indexed_at:{source}")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            logger.warning("store: unparseable indexed_at for source=%s: %r", source, raw)
            return None

    def pipeline(self, source: str) -> Optional[str]:
        """The pipeline fingerprint this source's chunks were made with, or None if never recorded."""
        if not self.exists():
            return None
        with self._connect() as con:
            return self._meta_get(con, f"pipeline:{source}")

    def set_pipeline(self, source: str, fingerprint: str) -> None:
        with self._connect() as con:
            self._meta_set(con, f"pipeline:{source}", fingerprint)
            con.commit()

    def sources(self) -> list[str]:
        """Source ids that have anything indexed (manifest entries)."""
        if not self.exists():
            return []
        with self._connect() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT source FROM files ORDER BY source").fetchall()]

    def clear_all(self) -> int:
        """Drop every source, the vector table and the model/dim identity. Returns rows deleted.
        Used for a model switch: the vector table's dimension is fixed, so the store must be
        empty before the first vector of a new model is written."""
        if not self.exists():
            return 0
        with self._connect() as con:
            n = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM files")
            con.execute("DELETE FROM meta WHERE key LIKE 'indexed_at:%'")
            self._reset_vectors(con)
            self._meta_set(con, "deleted_since_vacuum", "0")
            con.commit()
        self.vacuum(reason="model switch")
        return n

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

    def existing_chunk_ids(self, source: str, rel_path: str) -> set[str]:
        """Chunk ids the store holds for one file — lets a caller decide what to embed
        before calling sync_file (so many files' new chunks can be embedded in one go)."""
        if not self.exists():
            return set()
        with self._connect() as con:
            return {cid for (cid,) in con.execute(
                "SELECT chunk_id FROM chunks WHERE source = ? AND rel_path = ?", (source, rel_path)).fetchall()}

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
                # VACUUM alone leaves FTS5's segments as they are; merging them first is what
                # actually returns a deleted chunk's index space.
                if self._has_table(con, "chunks_fts"):
                    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
                    con.commit()
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
        return [ChunkRow(chunk_id=cid, doc=doc, metadata=json.loads(mj), distance=float(dist),
                         rowid=rid, source=source)
                for rid, dist, cid, doc, mj in rows]

    def keyword_search(self, sources: list[str], match: str, k: int,
                       content_type: Optional[str] = None) -> list[ChunkRow]:
        """Top-k chunks by BM25 for an FTS5 MATCH expression (see lexical.build_match), over
        several sources at once: one index means one IDF, so BM25 is comparable across them.

        The match runs first and is materialized, then rows are joined and filtered by the
        exact source and content_type. An inner LIMIT would let stronger out-of-scope matches
        fill the window and return a false empty (qmd #922). The source column filter keeps
        that match set small; it can admit a source whose id contains another's tokens, which
        the exact check below removes.
        """
        if not self.exists() or not sources:
            return []
        expr = f"doc : ({match})"
        scope = source_filter(sources)
        if scope:
            expr = f"{scope} AND {expr}"
        marks = ",".join("?" * len(sources))
        sql = ("WITH m AS MATERIALIZED (SELECT rowid, bm25(chunks_fts, 0.0, 1.0) AS score "
               "FROM chunks_fts WHERE chunks_fts MATCH ?) "
               "SELECT c.id, m.score, c.source, c.chunk_id, c.doc, c.meta_json "
               f"FROM m JOIN chunks c ON c.id = m.rowid WHERE c.source IN ({marks})")
        params: list = [expr, *sources]
        if content_type:
            sql += " AND c.content_type = ?"
            params.append(content_type)
        sql += " ORDER BY m.score ASC, c.id ASC LIMIT ?"
        params.append(int(k))
        with self._connect() as con:
            if not self._has_table(con, "chunks_fts"):
                raise StoreError(
                    f"store at {self.path} has no keyword index yet (schema 1). Any index run, or "
                    f"opening it with a current basic-kb, builds it; or search with --retrieval vector.")
            rows = con.execute(sql, params).fetchall()
        return [ChunkRow(chunk_id=cid, doc=doc, metadata=json.loads(mj), rowid=rid, source=src,
                         bm25=float(score))
                for rid, score, src, cid, doc, mj in rows]

    def cosine_distances(self, rowids: list[int], query: list[float]) -> dict[int, float]:
        """Cosine distance from `query` to the stored vectors of these chunks, so a hit found
        only by keyword still carries a similarity comparable with vector hits."""
        if not rowids or not self.exists():
            return {}
        out: dict[int, float] = {}
        with self._connect() as con:
            if self._meta_get(con, "dim") is None:
                return {}
            for i in range(0, len(rowids), 500):
                part = rowids[i:i + 500]
                marks = ",".join("?" * len(part))
                out.update(con.execute(
                    f"SELECT rowid, vec_distance_cosine(embedding, ?) FROM vec WHERE rowid IN ({marks})",
                    [_pack(query), *part]).fetchall())
        return {int(r): float(d) for r, d in out.items()}
