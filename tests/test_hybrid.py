"""Hybrid search end to end: the keyword index kept in step with chunks, the schema-1 upgrade,
scoping, and how vector and keyword lists are fused. Oracle patterns: qmd's scoped-search
false-empty fix (#922) and its trigger bugs (#792), qdrant-client's fusion cases (SPEC.md)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from basic_kb.client import RemoteKnowledgeBase
from basic_kb.core import KnowledgeBase
from basic_kb.errors import StoreError
from basic_kb.fusion import rrf
from basic_kb.lexical import build_match
from basic_kb.store import SqliteVecStore, VacuumPolicy

from .fakes import FakeEmbedder

E = FakeEmbedder()
CH = dict(chunk_size=400, overlap=40, min_chunk=20)


def put(store: SqliteVecStore, source: str, rel: str, docs: list[str], ct: str = "") -> None:
    ids = [f"{rel}::{hashlib.sha1(d.encode()).hexdigest()[:8]}" for d in docs]
    metas = [{"content_type": ct, "rel_path": rel} for _ in docs]
    store.sync_file(source, rel, "h-" + "".join(docs)[:20], ids, docs, metas, embed=E.embed, model_id=E.model_id)


def kw(store, sources, query, k=10, ct=None) -> list[str]:
    return [f"{r.source}/{r.metadata['rel_path']}" for r in store.keyword_search(sources, build_match(query), k, ct)]


def integrity(store) -> None:
    con = sqlite3.connect(store.path)
    con.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES ('integrity-check', 1)")
    con.close()


@pytest.fixture
def store(tmp_path: Path) -> SqliteVecStore:
    return SqliteVecStore(tmp_path / "store", vacuum=VacuumPolicy(enabled=False))


# --- the keyword index follows the chunks --------------------------------------------------

def test_index_follows_inserts_replacements_and_deletes(store):
    put(store, "notes", "a.md", ["the grinder broke on monday"])
    put(store, "notes", "b.md", ["kettle descale with citric acid"])
    assert kw(store, ["notes"], "grinder") == ["notes/a.md"]
    put(store, "notes", "a.md", ["the espresso machine broke instead"])   # chunk replaced
    assert kw(store, ["notes"], "grinder") == []
    assert kw(store, ["notes"], "espresso") == ["notes/a.md"]
    store.remove_files("notes", ["a.md"])
    assert kw(store, ["notes"], "espresso") == []
    store.clear_source("notes")
    assert kw(store, ["notes"], "kettle") == []
    integrity(store)


def test_clear_all_empties_the_keyword_index(store):
    put(store, "notes", "a.md", ["grinder"])
    store.clear_all()
    put(store, "notes", "b.md", ["kettle"])
    assert kw(store, ["notes"], "grinder") == [] and kw(store, ["notes"], "kettle") == ["notes/b.md"]
    integrity(store)


def test_a_schema_1_store_builds_its_index_from_stored_text(store):
    put(store, "notes", "a.md", ["the grinder broke on monday"])
    con = sqlite3.connect(store.path)                  # turn it back into a version-1 store
    for t in ("chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"):
        con.execute(f"DROP TRIGGER {t}")
    con.execute("DROP TABLE chunks_fts")
    con.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    con.commit()
    con.close()
    assert store.keyword_index_ready() is False
    with pytest.raises(StoreError, match="no keyword index"):
        kw(store, ["notes"], "grinder")
    calls = len(E.calls)
    store.ensure_schema()
    assert store.keyword_index_ready() and kw(store, ["notes"], "grinder") == ["notes/a.md"]
    assert len(E.calls) == calls                       # rebuilt from text, nothing re-embedded
    con = sqlite3.connect(store.path)
    assert con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone() == ("2",)


def test_scope_is_the_exact_source_and_content_type(store):
    put(store, "remote-box", "a.md", ["traefik certificate renewal"], ct="chat")
    put(store, "remote-box-old", "a.md", ["traefik certificate renewal"], ct="chat")
    put(store, "notes", "n.md", ["traefik certificate renewal"], ct="note")
    assert kw(store, ["remote-box"], "traefik") == ["remote-box/a.md"]
    assert sorted(kw(store, ["remote-box", "notes"], "traefik")) == ["notes/n.md", "remote-box/a.md"]
    assert kw(store, ["remote-box", "remote-box-old", "notes"], "traefik", ct="note") == ["notes/n.md"]


def test_a_scoped_search_sees_past_stronger_out_of_scope_matches(store):
    for i in range(30):
        put(store, "big", f"{i}.md", ["zebra zebra zebra stripes"])
    put(store, "small", "s.md", ["one zebra among many other words in a longer sentence here"])
    assert kw(store, ["small"], "zebra", k=1) == ["small/s.md"]


def test_cosine_distances_read_stored_vectors(store):
    put(store, "notes", "a.md", ["grinder broke"])
    (row,) = store.keyword_search(["notes"], build_match("grinder"), 5)
    d = store.cosine_distances([row.rowid], E.embed(["grinder broke"])[0])
    assert d[row.rowid] == pytest.approx(0.0, abs=1e-6)
    assert store.cosine_distances([], [0.0] * 64) == {}


def test_vacuum_merges_the_keyword_index(tmp_path):
    s = SqliteVecStore(tmp_path / "s", vacuum=VacuumPolicy(enabled=True, min_deleted=1, deleted_fraction=0.0))
    for i in range(5):
        put(s, "notes", f"{i}.md", [f"doc {i} about grinders"])
    s.remove_files("notes", ["0.md", "1.md"])
    assert s.vacuum() is True
    assert len(kw(s, ["notes"], "grinders")) == 3
    integrity(s)


# --- fusion inside search ------------------------------------------------------------------

def keys(hits) -> list[str]:
    return [h.doc for h in hits]


# A corpus on which the two retrievers disagree. "zebra" is in almost every chunk, so BM25
# gives it ~no weight while the bag-of-words vectors love the zebra-heavy chunks. "quartz"
# sits once in a long, diluted chunk: last for vectors, first for keywords.
DISAGREE = {
    "a.md": "zebra zebra zebra zebra",
    **{f"d{i}.md": f"zebra {w}" for i, w in enumerate(["lion", "tiger", "otter", "heron", "bison", "moose"])},
    "b.md": "quartz alpha beta gamma delta epsilon zeta eta theta iota",
}


@pytest.fixture
def disagree(tmp_path):
    from basic_kb.sources import MarkdownSource
    from .conftest import write_tree
    write_tree(tmp_path / "zoo", DISAGREE)
    src = MarkdownSource("zoo", tmp_path / "zoo", chunker="recursive")
    kb = KnowledgeBase(embedder=FakeEmbedder(), store_dir=tmp_path / "store")
    kb.index(src, chunk_size=400, overlap=0, min_chunk=1)
    return kb, src


def test_hybrid_fuses_two_lists_that_disagree(disagree):
    kb, src = disagree
    q, every = ["zebra quartz"], dict(n=50, cand_min=50)
    vec = keys(kb.search([src], q, retrieval="vector", **every))
    lex = keys(kb.search([src], q, retrieval="keyword", **every))
    assert vec[-1] == lex[0] == DISAGREE["b.md"]               # the premise: they disagree
    for k in (0, 20, 500):
        hyb = keys(kb.search([src], q, retrieval="hybrid", rrf_k=k, **every))
        assert hyb == [key for key, _ in rrf([vec, lex], k=k)]
    lifted = keys(kb.search([src], q, retrieval="hybrid", rrf_k=0, **every)).index(DISAGREE["b.md"])
    assert lifted < len(vec) - 1                               # fusion moved the keyword hit up


def test_hybrid_is_the_rrf_of_the_vector_and_keyword_lists(indexed_kb, sources):
    q = ["burr grinder kettle receipts"]
    every = dict(n=50, cand_min=50)
    vec = keys(indexed_kb.search(sources, q, retrieval="vector", **every))
    lex = keys(indexed_kb.search(sources, q, retrieval="keyword", **every))
    hyb = indexed_kb.search(sources, q, retrieval="hybrid", rrf_k=20, **every)
    assert keys(hyb) == [k for k, _ in rrf([vec, lex], k=20)]
    both = [h for h in hyb if set(h.matched_by) == {"vector", "keyword"}]
    assert both and all(h.fused_score is not None for h in hyb)


def test_multi_query_fuses_by_rank_not_by_raw_score(indexed_kb, notes):
    qs = ["coffee grinder", "tax receipts"]
    per_query = [keys(indexed_kb.search([notes], [q], retrieval="vector", n=50, cand_min=50)) for q in qs]
    fused = keys(indexed_kb.search([notes], qs, retrieval="vector", n=50, cand_min=50))
    assert fused == [k for k, _ in rrf(per_query, k=20)]


def test_keyword_only_hits_still_carry_cosine(indexed_kb, notes):
    hits = indexed_kb.search([notes], ["gooseneck"], retrieval="keyword")
    assert hits and all(set(h.matched_by) == {"keyword"} for h in hits)
    assert all(0.0 < h.score <= 1.0 for h in hits)
    assert "gooseneck" in hits[0].doc.lower()


def test_a_query_with_no_keyword_terms_falls_back_to_vectors(indexed_kb, notes):
    hits = indexed_kb.search([notes], ["the a of"], retrieval="hybrid")
    assert hits and all(set(h.matched_by) <= {"vector", "keyword"} for h in hits)
    assert indexed_kb.search([notes], ["!!! ???"], retrieval="keyword") == []


def test_unknown_retrieval_is_refused(indexed_kb, notes):
    with pytest.raises(ValueError, match="retrieval must be one of"):
        indexed_kb.search([notes], ["x"], retrieval="sparse")


def test_search_builds_the_index_of_a_schema_1_store(kb, sources):
    kb.index(sources[0], **CH)
    con = sqlite3.connect(kb.store.path)
    for t in ("chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"):
        con.execute(f"DROP TRIGGER {t}")
    con.execute("DROP TABLE chunks_fts")
    con.commit()
    con.close()
    assert kb.search([sources[0]], ["grinder"], retrieval="keyword")


def test_reranker_sees_the_fused_pool(config, sources, indexed_kb):
    from .fakes import FakeReranker
    rr = FakeReranker()
    kb = KnowledgeBase(embedder=FakeEmbedder(), store_dir=config.store_dir, reranker=rr)
    hits = kb.search(sources, ["burr grinder"], n=2, cand_min=6, cand_max=6)
    assert len(hits) == 2 and all(h.rerank_score is not None for h in hits)
    assert all(h.fused_score is not None and h.matched_by for h in hits)


# --- the surfaces --------------------------------------------------------------------------

def test_retrieval_travels_over_http(served, sources, notes):
    remote = RemoteKnowledgeBase(served.url)
    hits = remote.search([notes], ["gooseneck"], retrieval="keyword")
    assert hits and all(h.matched_by == {"keyword": h.matched_by["keyword"]} for h in hits)
    default = remote.search([notes], ["gooseneck kettle"])
    assert default and default[0].fused_score is not None


def test_cli_flag_and_rendering(fake_provider, instance, capsys):
    from basic_kb import cli
    cfg = str(instance / "basic-kb.yaml")
    cli.main(["index", "--json", "--config", cfg])
    capsys.readouterr()
    cli.main(["search", "gooseneck", "--retrieval", "keyword", "--source", "notes", "--config", cfg])
    out = capsys.readouterr().out
    assert "(keyword #1)" in out
    cli.main(["search", "gooseneck", "--retrieval", "keyword", "--json", "--config", cfg])
    hit = json.loads(capsys.readouterr().out)["hits"][0]
    assert set(hit["matched_by"]) == {"keyword"} and hit["fused_score"] > 0


def test_config_validates_search_retrieval(instance):
    from basic_kb.config import load_config
    path = instance / "basic-kb.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "search:\n  retrieval: sparse\n", encoding="utf-8")
    with pytest.raises(ValueError, match="search.retrieval"):
        load_config(path)
