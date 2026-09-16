"""SqliteVecStore: the real sqlite-vec store on disk, with vectors from the fake embedder."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from basic_kb.errors import StoreError
from basic_kb.store import SqliteVecStore, VacuumPolicy

from .fakes import FakeEmbedder

E = FakeEmbedder()


def embed(texts):
    return E.embed(texts)


def put(store: SqliteVecStore, source: str, rel: str, docs: list[str], *, ct: str = "", h: str = "h1",
        model: str = E.model_id):
    """Write one file's chunks with content-derived ids. Returns (embedded, reused)."""
    ids = [f"{rel}::{i}-{hashlib.sha1(d.encode()).hexdigest()[:8]}" for i, d in enumerate(docs)]
    metas = [{"content_type": ct, "position": str(i), "rel_path": rel} for i in range(len(docs))]
    return store.sync_file(source, rel, h, ids, docs, metas, embed=embed, model_id=model)


@pytest.fixture
def store(tmp_path: Path) -> SqliteVecStore:
    return SqliteVecStore(tmp_path / "store", vacuum=VacuumPolicy(enabled=False))


# --- empty store ---------------------------------------------------------------------------------

def test_empty_store_reads_are_safe(store):
    assert store.exists() is False
    assert store.manifest("a") == {}
    assert store.count("a") == 0
    assert store.chars("a") == 0
    assert store.knn("a", [0.0] * 64, 5) == []
    assert store.model_info() is None
    assert store.sources() == []
    assert store.source_indexed("a") is False
    assert store.indexed_at("a") is None
    assert store.all_metadata("a") == []
    assert store.existing_chunk_ids("a", "x") == set()
    assert store.remove_files("a", ["x"]) == 0
    assert store.clear_source("a") == 0
    assert store.clear_all() == 0
    assert store.vacuum() is False
    store.check_model("anything")     # nothing stored yet: no complaint


# --- writes ------------------------------------------------------------------------------------------

def test_first_write_creates_store_and_records_model(store):
    n_new, n_kept = put(store, "a", "f.md", ["alpha beta", "gamma delta"])
    assert (n_new, n_kept) == (2, 0)
    assert store.exists()
    assert store.model_info() == (E.model_id, 64)
    assert store.manifest("a") == {"f.md": "h1"}
    assert store.count("a") == 2
    assert store.chars("a") == len("alpha beta") + len("gamma delta")
    assert store.source_indexed("a") is True
    assert store.sources() == ["a"]
    assert len(store.existing_chunk_ids("a", "f.md")) == 2


def test_unchanged_chunks_are_reused_not_embedded(store):
    put(store, "a", "f.md", ["alpha beta", "gamma delta"])
    E.calls.clear()
    assert put(store, "a", "f.md", ["alpha beta", "gamma delta"], h="h2") == (0, 2)
    assert E.calls == []                       # nothing embedded
    assert store.manifest("a") == {"f.md": "h2"}   # hash still refreshed


def test_changed_chunk_replaces_only_itself(store):
    put(store, "a", "f.md", ["alpha beta", "gamma delta"])
    assert put(store, "a", "f.md", ["alpha beta", "gamma CHANGED"]) == (1, 1)
    assert store.count("a") == 2
    ids = store.existing_chunk_ids("a", "f.md")
    assert len(ids) == 2 and any(i.startswith("f.md::1-") for i in ids)
    docs = {m["position"] for m in store.all_metadata("a")}
    assert docs == {"0", "1"}


def test_content_type_change_moves_vector_row(store):
    put(store, "a", "f.md", ["alpha beta"], ct="x")
    assert len(store.knn("a", embed(["alpha beta"])[0], 5, content_type="x")) == 1
    put(store, "a", "f.md", ["alpha beta"], ct="y")
    assert store.knn("a", embed(["alpha beta"])[0], 5, content_type="x") == []
    assert len(store.knn("a", embed(["alpha beta"])[0], 5, content_type="y")) == 1


def test_empty_chunk_list_drops_rows_but_tracks_file(store):
    put(store, "a", "f.md", ["alpha beta"])
    assert put(store, "a", "f.md", [], h="h9") == (0, 0)
    assert store.count("a") == 0
    assert store.manifest("a") == {"f.md": "h9"}
    assert store.source_indexed("a") is True


def test_remove_files_prunes_chunks_vectors_and_manifest(store):
    put(store, "a", "f.md", ["alpha beta"])
    put(store, "a", "g.md", ["gamma delta"])
    assert store.remove_files("a", ["f.md"]) == 1
    assert store.manifest("a") == {"g.md": "h1"}
    assert store.count("a") == 1
    assert [r.doc for r in store.knn("a", embed(["alpha beta"])[0], 5)] == ["gamma delta"]


def test_clear_source_leaves_other_sources(store):
    put(store, "a", "f.md", ["alpha beta"])
    put(store, "b", "f.md", ["gamma delta"])
    assert store.clear_source("a") == 1
    assert store.sources() == ["b"]
    assert store.model_info() == (E.model_id, 64)      # b still holds vectors


def test_clearing_the_last_source_forgets_the_model(store):
    put(store, "a", "f.md", ["alpha beta"])
    store.clear_source("a")
    assert store.model_info() is None
    # A different model may now be written.
    other = FakeEmbedder(dim=16, model_id="other")
    ids, docs = ["f.md::0"], ["x"]
    store.sync_file("a", "f.md", "h", ids, docs, [{}], embed=other.embed, model_id="other")
    assert store.model_info() == ("other", 16)


def test_clear_all_resets_everything(store):
    put(store, "a", "f.md", ["alpha beta"])
    put(store, "b", "f.md", ["gamma delta"])
    store.set_indexed_at("a", 123.0)
    assert store.clear_all() == 2
    assert store.sources() == [] and store.count("a") == 0
    assert store.model_info() is None
    assert store.indexed_at("a") is None


# --- model guard --------------------------------------------------------------------------------------

def test_check_model_refuses_a_different_model(store):
    put(store, "a", "f.md", ["alpha beta"])
    store.check_model(E.model_id)
    with pytest.raises(StoreError, match="built with embedding model"):
        store.check_model("other")


def test_write_with_a_different_model_while_vectors_exist_is_refused(store):
    put(store, "a", "f.md", ["alpha beta"])
    other = FakeEmbedder(dim=16, model_id="other")
    with pytest.raises(StoreError, match="other sources still hold"):
        store.sync_file("b", "g.md", "h", ["g.md::0"], ["x"], [{}], embed=other.embed, model_id="other")


# --- knn ------------------------------------------------------------------------------------------------

def test_knn_ranks_exact_text_first_and_respects_k(store):
    put(store, "a", "f.md", ["alpha beta", "gamma delta", "epsilon zeta"])
    rows = store.knn("a", embed(["gamma delta"])[0], 2)
    assert len(rows) == 2
    assert rows[0].doc == "gamma delta"
    assert rows[0].distance == pytest.approx(0.0, abs=1e-6)
    assert rows[0].chunk_id.startswith("f.md::1-")
    assert rows[0].metadata["position"] == "1"


def test_knn_is_partitioned_by_source(store):
    put(store, "a", "f.md", ["alpha beta"])
    put(store, "b", "f.md", ["alpha beta"])
    assert len(store.knn("a", embed(["alpha beta"])[0], 10)) == 1


# --- timestamps -------------------------------------------------------------------------------------------

def test_indexed_at_round_trip(store):
    put(store, "a", "f.md", ["alpha beta"])
    assert store.indexed_at("a") is None
    store.set_indexed_at("a", 1700000000.5)
    assert store.indexed_at("a") == 1700000000.5
    store.set_indexed_at("a")
    assert store.indexed_at("a") > 1700000000.5


def test_indexed_at_unparseable_reads_as_none(store):
    put(store, "a", "f.md", ["alpha beta"])
    with sqlite3.connect(store.path) as con:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('indexed_at:a', 'garbage')")
    assert store.indexed_at("a") is None


# --- vacuum -------------------------------------------------------------------------------------------------

def test_vacuum_stats_and_policy(tmp_path: Path):
    store = SqliteVecStore(tmp_path / "s", vacuum=VacuumPolicy(enabled=True, deleted_fraction=0.2, min_deleted=1))
    put(store, "a", "f.md", ["alpha beta", "gamma delta", "epsilon zeta"])
    assert store.vacuum_stats() == {"live": 3, "deleted_since_vacuum": 0, "fraction": 0.0, "would_vacuum": False}
    store.remove_files("a", ["f.md"])            # 3 deleted of 0 live -> vacuumed, counter reset
    assert store.vacuum_stats()["deleted_since_vacuum"] == 0


def test_vacuum_disabled_keeps_counting(store):
    put(store, "a", "f.md", ["alpha beta", "gamma delta"])
    put(store, "a", "g.md", ["epsilon zeta"])
    store.remove_files("a", ["g.md"])
    st = store.vacuum_stats()
    assert st["deleted_since_vacuum"] == 1 and st["would_vacuum"] is False
    assert store.vacuum(reason="test") is True
    assert store.vacuum_stats()["deleted_since_vacuum"] == 0


# --- misc reads -----------------------------------------------------------------------------------------------

def test_chunk_ids_by_file_and_subset(store):
    put(store, "a", "f.md", ["alpha beta", "gamma delta"])
    put(store, "a", "g.md", ["epsilon zeta"])
    by_file = store.chunk_ids_by_file("a")
    assert set(by_file) == {"f.md", "g.md"} and len(by_file["f.md"]) == 2
    assert set(store.chunk_ids_by_file("a", ["g.md"])) == {"g.md"}
