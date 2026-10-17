"""Keyword query building: the MATCH strings, and that each one finds what it should in a
real FTS5 index. Cases follow the oracles' tests — qmd test/store-fts-separator-queries,
sqlite-utils test_quote_fts_query, kb TestFtsEscape (see SPEC.md)."""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from basic_kb.lexical import MAX_TERMS, build_match, source_filter
from basic_kb.store import SqliteVecStore, VacuumPolicy

from .fakes import FakeEmbedder

E = FakeEmbedder()

DOCS = {
    "a.md": "The flash-and-forget gateway scaffold landed in PIO-1384, see src/lib/provision.ts.",
    "b.md": "cloud-init runs apply_secrets from __init__ before the 2026.4.10 release of @tobilu/qmd.",
    "c.md": "ECONNREFUSED when traefik retried the upstream; deployed a fix with --switch-model.",
    "d.md": "Kaffe og mødet om budgettet blev flyttet til café Ørsted.",
}


# --- the MATCH string ------------------------------------------------------------------------

@pytest.mark.parametrize("query, expected", [
    ("hello", '"hello"*'),
    ("hello world", '"hello"* OR "world"*'),
    ("what's the cost?", '"cost"*'),                       # stopwords and 1-char parts drop
    ("cat's toy", '"cat s" OR "toy"*'),                    # a contraction with a real word stays
    ("the a", '"the"* OR "a"'),                            # ...unless nothing else remains
    ("cache Cache cache api", '"cache"* OR "api"*'),       # case-folded, deduped
    ("PIO-1384", '"pio 1384"'),                            # separator -> phrase of its parts
    ("src/lib/provision.ts", '"src lib provision ts"'),
    ("--switch-model", '"switch model"'),
    ("apply_secrets", '"apply secrets"'),                  # `_` is a separator to unicode61
    ("__init__", '"init"*'),
    ('"gateway scaffold" deploy', '"gateway scaffold" OR "deploy"*'),
    ('cat"', '"cat"*'),                                    # unbalanced quote balanced
    ('"cat dog" "fish', '"cat dog" OR "fish"'),
    ("a OR NOT b", '"not"*'),                              # operators are just words
    ("db is up", '"db" OR "up"'),                          # short parts match whole, no prefix
])
def test_build_match(query, expected):
    assert build_match(query) == expected


@pytest.mark.parametrize("query", ["", "   ", '"', '""', "!@#$%", "- _ . /"])
def test_nothing_searchable_gives_none(query):
    assert build_match(query) is None


def test_runaway_query_is_capped():
    match = build_match(" ".join(f"term{i}" for i in range(MAX_TERMS + 10)))
    assert match.count(" OR ") == MAX_TERMS - 1


def test_source_filter():
    assert source_filter(["notes"]) == 'source : ("notes")'
    assert source_filter(["remote-box", "work-macos"]) == 'source : ("remote box" OR "work macos")'
    assert source_filter([]) == ""


# --- against a real index --------------------------------------------------------------------

@pytest.fixture
def store(tmp_path) -> SqliteVecStore:
    s = SqliteVecStore(tmp_path / "store", vacuum=VacuumPolicy(enabled=False))
    for rel, doc in DOCS.items():
        cid = f"{rel}::{hashlib.sha1(doc.encode()).hexdigest()[:8]}"
        s.sync_file("notes", rel, "h", [cid], [doc], [{"rel_path": rel}], embed=E.embed, model_id=E.model_id)
    return s


def hits(store, query) -> list[str]:
    match = build_match(query)
    return [r.metadata["rel_path"] for r in store.keyword_search(["notes"], match, 10)] if match else []


@pytest.mark.parametrize("query, rel", [
    ('"flash-and-forget"', "a.md"), ("flash-and-forget", "a.md"),
    ('"PIO-1384"', "a.md"), ("pio-1384", "a.md"), ('"src/lib/provision.ts"', "a.md"),
    ('"cloud-init"', "b.md"), ("2026.4.10", "b.md"), ('"@tobilu/qmd"', "b.md"),
    ("apply_secrets", "b.md"), ('"apply_secrets"', "b.md"), ("__init__", "b.md"),
    ("econnrefused", "c.md"), ("econnref", "c.md"), ("deploy", "c.md"), ("--switch-model", "c.md"),
    ("cafe", "d.md"),           # diacritics folded: é -> e (ø is a letter of its own, not folded)
])
def test_terms_find_their_chunk(store, query, rel):
    assert hits(store, query) == [rel]


def test_a_quoted_phrase_must_be_in_order(store):
    assert hits(store, '"flash-and-forget gateway scaffold"') == ["a.md"]
    assert hits(store, '"scaffold flash-and-forget"') == []
    assert hits(store, '"PIO-9999"') == []


def test_or_ranks_the_chunk_holding_more_terms_first(store):
    assert hits(store, "traefik gateway upstream retried") == ["c.md", "a.md"]


def test_no_input_reaches_fts5_as_syntax(store):
    for query in ['foo-bar', 'col:val', '"', '*', 'NEAR(a b)', '^foo', "cat's", 'a AND (b', 'x OR']:
        match = build_match(query)
        if match:
            store.keyword_search(["notes"], match, 5)          # must not raise
    with pytest.raises(sqlite3.OperationalError):              # the raw strings would
        con = sqlite3.connect(store.path)
        con.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("col:val",)).fetchall()
