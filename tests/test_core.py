"""KnowledgeBase: index, incremental behaviour, guard, search, status, scan, info."""
from __future__ import annotations

from pathlib import Path

import pytest

from basic_kb.core import KnowledgeBase, cores_to_threads
from basic_kb.errors import IndexNotFound, MassChangeRefused, QueryFailed, StoreError
from basic_kb.sources import MarkdownSource

from .conftest import write_tree
from .fakes import FailingReranker, FakeEmbedder, FakeReranker
from basic_kb.embedders import EMBEDDER_PROVIDERS
from basic_kb.rerankers import RERANKER_TYPES

CH = dict(chunk_size=400, overlap=40, min_chunk=20)


def many_source(root: Path, n: int = 6) -> MarkdownSource:
    """A source with n files, each long enough to chunk, for guard tests (floor is 5 files)."""
    write_tree(root, {f"n{i}.md": f"# Note {i}\n\nThis is note number {i} with enough words to make a chunk.\n"
                      for i in range(n)})
    return MarkdownSource("many", root, chunker="recursive")


def rewrite_all(src: MarkdownSource, suffix: str = " edited") -> None:
    for f in src.get_files():
        f.write_text(f.read_text(encoding="utf-8") + suffix + "\n", encoding="utf-8")


# --- index -------------------------------------------------------------------------------------------

def test_first_index_counts(kb, notes, embedder):
    r = kb.index(notes, **CH)
    assert not r.aborted and r.errors == []
    assert r.files_on_disk == 4                # drafts/ excluded
    assert (r.added, r.updated, r.unchanged, r.empty, r.pruned) == (3, 0, 0, 1, 0)
    assert r.embedded == 3
    assert r.chunks_embedded == r.total_chunks == 6
    assert r.chunks_reused == 0
    assert kb.store.indexed_at("notes") is not None
    assert set(kb.store.manifest("notes")) == {"coffee.md", "taxes.md", "sub/travel.md", "stub.md"}
    assert all(len(batch) <= KnowledgeBase.WAVE_CHUNKS for batch in embedder.calls)


def test_second_index_is_a_no_op(kb, notes, embedder):
    kb.index(notes, **CH)
    embedder.calls.clear()
    r = kb.index(notes, **CH)
    assert (r.added, r.updated, r.unchanged, r.chunks_embedded) == (0, 0, 4, 0)
    assert embedder.calls == []


def test_appending_re_embeds_only_new_chunks(kb, notes):
    kb.index(notes, **CH)
    p = notes.directory / "coffee.md"
    p.write_text(p.read_text(encoding="utf-8") + "\n## Storage\n\nKeep beans in an airtight jar away from light.\n",
                 encoding="utf-8")
    r = kb.index(notes, **CH)
    assert (r.added, r.updated, r.unchanged) == (0, 1, 3)
    assert r.chunks_embedded == 1
    assert r.chunks_reused == 3
    assert r.total_chunks == 7


def test_deleted_file_is_pruned(kb, notes):
    kb.index(notes, **CH)
    (notes.directory / "taxes.md").unlink()
    r = kb.index(notes, **CH)
    assert r.pruned == 1 and r.total_chunks == 4
    assert "taxes.md" not in kb.store.manifest("notes")


def test_progress_callback_receives_lines(kb, notes):
    lines: list[str] = []
    kb.index(notes, on_progress=lines.append, **CH)
    assert any("Indexing Test Notes" in l for l in lines)
    assert any("[1/4]" in l for l in lines)
    assert any(l.startswith("\nDone [notes]") for l in lines)


def test_force_re_embeds_everything(kb, notes, embedder):
    kb.index(notes, **CH)
    r = kb.index(notes, force=True, **CH)
    assert r.chunks_embedded == 6 and r.chunks_reused == 0
    assert r.added == 3                         # cleared first, so every file counts as new


def test_limit_caps_files_and_never_prunes(kb, notes):
    r = kb.index(notes, limit=1, **CH)
    assert r.limited_to == 1 and r.added + r.empty == 1 and r.files_on_disk == 4
    kb.index(notes, **CH)
    (notes.directory / "taxes.md").unlink()
    r = kb.index(notes, limit=2, **CH)
    assert r.pruned == 0
    assert "taxes.md" in kb.store.manifest("notes")


def test_parse_error_is_collected_not_raised(kb, notes, monkeypatch):
    real = notes.parse_file

    def flaky(path):
        if path.name == "taxes.md":
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad byte")
        return real(path)

    monkeypatch.setattr(notes, "parse_file", flaky)
    r = kb.index(notes, **CH)
    assert [e.rel_path for e in r.errors] == ["taxes.md"]
    assert "UnicodeDecodeError" in r.errors[0].error
    assert r.added == 2
    assert "taxes.md" not in kb.store.manifest("notes")     # retried next run


def test_missing_source_directory(kb, tmp_path):
    r = kb.index(MarkdownSource("gone", tmp_path / "nowhere"), **CH)
    assert r.files_on_disk == 0 and not r.aborted
    assert "No files found" in r.abort_reason


# --- index_many (one run over several sources) ---------------------------------------------------------

def test_index_many_budget_and_per_source(kb, sources):
    results = kb.index_many(sources, limit=1, **CH)
    assert len(results) == 1 and results[0].limited_to == 1       # budget spent on the first source
    results = kb.index_many(sources, limit=1, limit_per_source=True, force=True, **CH)
    assert [r.source_id for r in results] == ["notes", "meetings"]


def test_index_many_switch_model_wipes_and_rebuilds(indexed_kb, config, sources):
    other = KnowledgeBase(FakeEmbedder(dim=16, model_id="other"), config.store_dir)
    with pytest.raises(StoreError, match="Model switch detected"):
        other.index_many(sources, **CH)
    lines: list[str] = []
    results = other.index_many(sources, switch_model=True, on_progress=lines.append, **CH)
    assert lines[0].startswith("Model switch 'fake-bow-64' -> 'other'")
    assert [r.source_id for r in results] == ["notes", "meetings"] and all(r.added for r in results)
    assert other.store.model_info() == ("other", 16)


def test_index_many_clears_freshness_nudges(indexed_kb, sources, config):
    from basic_kb.freshness import FreshnessSettings, FreshnessTracker
    tracker = FreshnessTracker(config.store_dir, FreshnessSettings())
    tracker._write({"notes": {"nudges": 3}, "meetings": {"nudges": 1}})
    indexed_kb.index_many(sources[:1], **CH)
    assert tracker.state() == {"meetings": {"nudges": 1}}


# --- the emptied-source gotcha (2026-08-27) ------------------------------------------------------------

def test_emptied_source_prunes_its_vectors(kb, notes):
    kb.index(notes, **CH)
    for f in notes.get_files():
        f.unlink()
    r = kb.index(notes, guard=False, **CH)
    assert r.pruned == 4 and r.total_chunks == 0
    assert kb.store.manifest("notes") == {}


def test_emptied_source_under_guard_refuses_unattended(kb, tmp_path):
    src = many_source(tmp_path / "many")
    kb.index(src, **CH)
    for f in src.get_files():
        f.unlink()
    r = kb.index(src, **CH)
    assert r.aborted and "Mass change" in r.abort_reason
    assert kb.store.count("many") == 6            # index left intact


# --- mass-change guard ---------------------------------------------------------------------------------

def test_guard_refuses_by_default(kb, tmp_path):
    src = many_source(tmp_path / "many")
    kb.index(src, **CH)
    before = kb.store.manifest("many")
    rewrite_all(src)
    r = kb.index(src, **CH)
    assert r.aborted is True
    assert "6 changed + 0 deleted of 6" in r.abort_reason
    assert kb.store.manifest("many") == before


def test_guard_asks_on_confirm(kb, tmp_path):
    src = many_source(tmp_path / "many")
    kb.index(src, **CH)
    rewrite_all(src)
    seen: list[MassChangeRefused] = []
    r = kb.index(src, on_confirm=lambda d: seen.append(d) or False, **CH)
    assert r.aborted and seen[0].changed == 6 and seen[0].fraction == 1.0
    r = kb.index(src, on_confirm=lambda d: True, **CH)
    assert not r.aborted and r.updated == 6


def test_guard_accepts_assume_yes_and_skips_when_disabled_or_forced(kb, tmp_path):
    src = many_source(tmp_path / "many")
    kb.index(src, **CH)
    rewrite_all(src, " a")
    assert not kb.index(src, assume_yes=True, **CH).aborted
    rewrite_all(src, " b")
    assert not kb.index(src, guard=False, **CH).aborted
    rewrite_all(src, " c")
    assert not kb.index(src, force=True, **CH).aborted


def test_guard_has_a_five_file_floor(kb, tmp_path):
    src = many_source(tmp_path / "many", n=4)
    kb.index(src, **CH)
    rewrite_all(src)                               # 100% churn, but only 4 files
    assert not kb.index(src, **CH).aborted


def test_guard_threshold(kb, tmp_path):
    src = many_source(tmp_path / "many", n=10)
    kb.index(src, **CH)
    for f in src.get_files()[:5]:                  # 50% churn, 5 files
        f.write_text("# changed\n\nEnough words here to still make a chunk after the edit.\n", encoding="utf-8")
    assert kb.index(src, guard_threshold=0.9, **CH).aborted is False
    for f in src.get_files()[:5]:
        f.write_text("# changed again\n\nEnough words here to still make a chunk after the edit.\n", encoding="utf-8")
    assert kb.index(src, guard_threshold=0.5, **CH).aborted is True


# --- model switch ----------------------------------------------------------------------------------------

def test_model_switch_refused_unless_accepted(indexed_kb, config, sources):
    other = KnowledgeBase(FakeEmbedder(dim=16, model_id="other"), config.store_dir)
    with pytest.raises(StoreError, match="Model switch detected"):
        other.prepare_model_switch(sources, accept=False)
    with pytest.raises(StoreError, match="missing: notes"):
        other.prepare_model_switch(sources[1:], accept=True)
    note = other.prepare_model_switch(sources, accept=True)
    assert "cleared" in note
    assert other.store.model_info() is None
    assert indexed_kb.prepare_model_switch(sources) is None       # matching model: nothing to do


def test_index_with_wrong_model_is_refused(indexed_kb, config, notes):
    other = KnowledgeBase(FakeEmbedder(dim=16, model_id="other"), config.store_dir)
    with pytest.raises(StoreError):
        other.index(notes, **CH)


# --- search -----------------------------------------------------------------------------------------------

def test_search_ranks_matching_chunk_first(indexed_kb, sources):
    hits = indexed_kb.search(sources, ["burr grinder gooseneck kettle"], n=3)
    assert 1 <= len(hits) <= 3
    assert "burr grinder" in hits[0].doc
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)
    assert hits[0].metadata["source"] in {"notes", "meetings"}
    assert "rel_path" in hits[0].metadata and "content_hash" in hits[0].metadata


def test_search_merges_across_sources(indexed_kb, sources):
    hits = indexed_kb.search(sources, ["coffee burr grinder"], n=10)
    assert {h.metadata["source"] for h in hits} == {"notes", "meetings"}


def test_search_respects_n_and_content_type(indexed_kb, notes):
    hits = indexed_kb.search([notes], ["receipts audit tax return"], n=1)
    assert len(hits) == 1
    admin = indexed_kb.search([notes], ["anything at all"], n=10, content_type_filter="admin")
    assert admin and all(h.metadata["content_type"] == "admin" for h in admin)


def test_multi_query_fuses_into_one_list(indexed_kb, notes, embedder):
    hits = indexed_kb.search([notes], ["coffee grinder", "tax receipts"], n=10)
    docs = [h.doc for h in hits]
    assert len(docs) == len(set(docs))            # deduplicated by chunk
    assert any("grinder" in d for d in docs) and any("receipts" in d for d in docs)
    assert embedder.query_calls[-2:] == [["coffee grinder"], ["tax receipts"]]


def test_offset_pages_through_one_ranked_list(indexed_kb, sources):
    """Page two continues where page one stopped, on the same ranking."""
    whole = indexed_kb.search(sources, ["coffee burr grinder"], n=6)
    assert len(whole) >= 4                                   # enough to page

    first = indexed_kb.search(sources, ["coffee burr grinder"], n=2)
    second = indexed_kb.search(sources, ["coffee burr grinder"], n=2, offset=2)
    assert [h.doc for h in first] == [h.doc for h in whole[:2]]
    assert [h.doc for h in second] == [h.doc for h in whole[2:4]]
    assert not ({h.doc for h in first} & {h.doc for h in second})


def test_offset_past_the_end_is_empty_and_a_negative_one_is_refused(indexed_kb, sources):
    assert indexed_kb.search(sources, ["coffee burr grinder"], n=5, offset=10_000) == []
    with pytest.raises(ValueError, match="offset must be zero or greater"):
        indexed_kb.search(sources, ["coffee"], offset=-1)


def test_offset_pages_each_group_separately(indexed_kb, notes):
    groups = indexed_kb.search_grouped([notes], ["coffee grinder", "tax receipts"], n=1, offset=1)
    assert [q for q, _ in groups] == ["coffee grinder", "tax receipts"]
    firsts = indexed_kb.search_grouped([notes], ["coffee grinder", "tax receipts"], n=1)
    for (_, page_two), (_, page_one) in zip(groups, firsts):
        assert [h.doc for h in page_two] != [h.doc for h in page_one]


def test_reranking_stops_at_the_ceiling_however_deep_the_offset(indexed_kb, sources):
    """The candidate pool is what costs money, so paging never widens it."""
    indexed_kb.reranker = rr = FakeReranker()
    indexed_kb.search(sources, ["coffee burr grinder"], n=2, offset=500,
                      cand_multiplier=3, cand_min=1, cand_max=4)
    assert rr.calls[-1][1] <= 4


def test_search_grouped_returns_one_block_per_query(indexed_kb, notes):
    groups = indexed_kb.search_grouped([notes], ["coffee grinder", "tax receipts"], n=2)
    assert [q for q, _ in groups] == ["coffee grinder", "tax receipts"]
    assert all(len(h) <= 2 for _, h in groups)


def test_search_without_store_raises(kb, notes):
    with pytest.raises(IndexNotFound, match="No index"):
        kb.search([notes], ["x"])


def test_search_with_no_indexed_source_raises_but_partial_works(kb, notes, meetings):
    kb.index(notes, **CH)
    with pytest.raises(IndexNotFound, match="meetings"):
        kb.search([meetings], ["x"])
    assert kb.search([notes, meetings], ["coffee"], n=3)     # one indexed source is enough


def test_query_failure_is_distinct_from_no_match(indexed_kb, notes, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(indexed_kb.store, "knn", boom)
    with pytest.raises(QueryFailed, match="disk on fire"):
        indexed_kb.search([notes], ["x"])


def test_search_with_wrong_model_is_refused(indexed_kb, config, notes):
    other = KnowledgeBase(FakeEmbedder(dim=16, model_id="other"), config.store_dir)
    with pytest.raises(StoreError):
        other.search([notes], ["x"])


def test_reranker_reorders_and_scores(config, sources, embedder):
    rr = FakeReranker()
    kb = KnowledgeBase(embedder, config.store_dir, reranker=rr)
    for s in sources:
        kb.index(s, **CH)
    hits = kb.search(sources, ["seven years receipts audit"], n=2)
    assert rr.calls[0][0] == "seven years receipts audit"
    assert rr.calls[0][1] == 8                     # every chunk was a candidate (cand_min 50 > 8)
    assert len(hits) == 2 and all(h.rerank_score is not None for h in hits)
    assert "receipts" in hits[0].doc
    assert hits == sorted(hits, key=lambda h: h.sort_key, reverse=True)


def test_rerank_candidates_param_caps_the_pool(config, sources, embedder):
    rr = FakeReranker()
    kb = KnowledgeBase(embedder, config.store_dir, reranker=rr)
    for s in sources:
        kb.index(s, **CH)
    kb.search(sources, ["coffee"], n=2, rerank_candidates=3)
    assert rr.calls[0][1] == 3


def test_failing_reranker_falls_back_unless_strict(config, sources, embedder):
    kb = KnowledgeBase(embedder, config.store_dir, reranker=FailingReranker())
    for s in sources:
        kb.index(s, **CH)
    hits = kb.search(sources, ["coffee"], n=2)
    assert hits and all(h.rerank_score is None for h in hits)
    with pytest.raises(RuntimeError, match="reranker down"):
        kb.search(sources, ["coffee"], n=2, strict_rerank=True)


# --- status / scan / info ------------------------------------------------------------------------------

def test_status_values(indexed_kb, sources):
    notes_st, meet_st = indexed_kb.status(sources)
    assert notes_st.indexed and notes_st.tracked and notes_st.directory_exists
    assert notes_st.chunks == 6 and notes_st.chars > 0 and notes_st.approx_tokens == notes_st.chars // 4
    assert notes_st.docs_with_chunks == 3 and notes_st.files_on_disk == 4
    assert notes_st.stale == 0 and notes_st.pending == 0
    assert notes_st.content_types == {"hobby": 4, "admin": 2}
    assert notes_st.date_min is None
    assert notes_st.indexed_at is not None
    assert (meet_st.date_min, meet_st.date_max) == ("2026-01-05", "2026-02-10")
    assert meet_st.model_id == "fake-bow-64"


def test_status_unindexed_source_and_missing_store(kb, notes, meetings):
    with pytest.raises(IndexNotFound):
        kb.status([notes])
    kb.index(notes, **CH)
    st = kb.status([meetings])[0]
    assert st.indexed is False and st.chunks == 0


def test_scan_tracks_new_updated_deleted(kb, notes):
    assert kb.scan(notes).tracked is False
    kb.index(notes, **CH)
    s = kb.scan(notes)
    assert (s.tracked, s.files_on_disk, s.unchanged, s.stale) == (True, 4, 4, 0)
    (notes.directory / "coffee.md").write_text("# Coffee\n\nA totally new body for the coffee note.\n", encoding="utf-8")
    (notes.directory / "new.md").write_text("# New\n\nA brand new note that has not been indexed.\n", encoding="utf-8")
    (notes.directory / "taxes.md").unlink()
    s = kb.scan(notes)
    assert (s.new, s.updated, s.deleted, s.unchanged) == (1, 1, 1, 2)
    assert s.stale == 3


def test_info_describes_sources_with_or_without_store(kb, indexed_kb, sources):
    inf = indexed_kb.info(sources, name="inst")
    assert inf.name == "inst" and inf.model_id == "fake-bow-64"
    assert inf.total_files == 6 and inf.total_chunks == 8
    n = inf.sources[0]
    assert (n.source_id, n.type, n.chunker, n.files, n.chunks, n.indexed) == ("notes", "markdown", "breadcrumb", 4, 6, True)
    assert n.description == "Markdown notes with frontmatter."
    assert n.chunks_per_file == 1.5
    fresh = KnowledgeBase(FakeEmbedder(), kb.store_dir / "empty")
    assert all(not s.indexed for s in fresh.info(sources).sources)


# --- preview (dry-run chunking) ------------------------------------------------------------------------

def test_preview_reports_chunks_without_touching_store_or_embedder(kb, notes, embedder):
    files = kb.preview(notes, **CH)
    assert [f.rel_path for f in files] == ["coffee.md", "stub.md", "sub/travel.md", "taxes.md"]
    coffee = files[0]
    assert coffee.body_chars > 0 and len(coffee.chunks) == 3 and not coffee.skipped
    assert coffee.chunks[1].breadcrumb == "# Coffee brewing > ## Ratios"
    assert coffee.chunks[1].text.startswith("# Coffee brewing > ## Ratios")
    stub = files[1]
    assert stub.chunks == [] and stub.body_chars > 0        # parsed, but nothing survived min_chunk
    assert embedder.calls == [] and not kb.store.exists()


def test_preview_limit_and_file_name(kb, notes):
    assert [f.rel_path for f in kb.preview(notes, limit=2, **CH)] == ["coffee.md", "stub.md"]
    assert [f.rel_path for f in kb.preview(notes, file_name="taxes.md", **CH)] == ["taxes.md"]
    assert kb.preview(notes, file_name="nope.md", **CH) == []


# --- reindex_paths (the watcher's write path) --------------------------------------------------------

def test_reindex_paths_handles_changed_unchanged_deleted_and_empty(kb, notes):
    kb.index(notes, **CH)
    stamp = kb.store.indexed_at("notes")
    coffee = notes.directory / "coffee.md"
    coffee.write_text(coffee.read_text(encoding="utf-8") + "\nAn extra line about coffee storage in jars.\n", encoding="utf-8")
    (notes.directory / "taxes.md").unlink()
    (notes.directory / "sub" / "travel.md").write_text("---\ntitle: T\n---\n\nx\n", encoding="utf-8")
    counts = kb.reindex_paths(notes, [coffee, notes.directory / "taxes.md", notes.directory / "stub.md",
                                      notes.directory / "sub" / "travel.md"], **CH)
    assert (counts.embedded, counts.pruned, counts.unchanged, counts.empty) == (1, 1, 1, 1)
    assert counts.chunks_embedded >= 1 and counts.changed
    assert kb.store.indexed_at("notes") >= stamp
    nothing = kb.reindex_paths(notes, [], **CH)
    assert nothing.embedded == 0 and not nothing.changed and nothing.summary() == "no change"


def test_stale_paths_lists_changed_and_deleted(kb, notes):
    kb.index(notes, **CH)
    assert kb.stale_paths(notes) == []
    (notes.directory / "coffee.md").write_text("# C\n\nchanged body with enough words in it.\n", encoding="utf-8")
    (notes.directory / "taxes.md").unlink()
    stale = {p.name for p in kb.stale_paths(notes)}
    assert stale == {"coffee.md", "taxes.md"}


# --- from_config (the composition root) ---------------------------------------------------------------

@pytest.fixture
def registries(monkeypatch):
    monkeypatch.setitem(EMBEDDER_PROVIDERS, "fake", lambda cfg, threads=None: FakeEmbedder(model_id=f"{cfg.embedding_model}/t{threads}"))
    monkeypatch.setitem(RERANKER_TYPES, "fake-rr", lambda model=None, **opts: FakeReranker())

    class NeedsKey(FakeReranker):
        def __init__(self, model=None, **opts):
            raise ValueError("FAKE_KEY not set")
    monkeypatch.setitem(RERANKER_TYPES, "needs-key", NeedsKey)


def test_from_config_builds_embedder_store_and_vacuum(config, registries):
    from dataclasses import replace
    cfg = replace(config, embedding={"provider": "fake"})
    kb = KnowledgeBase.from_config(cfg, threads=2)
    assert kb.embedder.model_id == "fake-bow-64/t2"
    assert kb.store_dir == config.store_dir
    assert kb.store.vacuum_policy.enabled is False           # from the fixture's vacuum block
    assert kb.reranker is None                               # config says none


def test_from_config_model_override(config, registries):
    from dataclasses import replace
    cfg = replace(config, embedding={"provider": "fake"})
    assert KnowledgeBase.from_config(cfg, model="other").embedder.model_id == "other/tNone"


def test_from_config_reranker_resolution(config, registries):
    from dataclasses import replace
    cfg = replace(config, embedding={"provider": "fake"}, reranker_type="fake-rr")
    assert isinstance(KnowledgeBase.from_config(cfg).reranker, FakeReranker)
    assert KnowledgeBase.from_config(cfg, reranker="none").reranker is None
    assert isinstance(KnowledgeBase.from_config(replace(cfg, reranker_type="none"), reranker="fake-rr").reranker, FakeReranker)


def test_from_config_unavailable_reranker_warns_or_raises(config, registries):
    from dataclasses import replace
    cfg = replace(config, embedding={"provider": "fake"}, reranker_type="needs-key")
    warnings: list[str] = []
    kb = KnowledgeBase.from_config(cfg, on_warning=warnings.append)
    assert kb.reranker is None
    assert warnings and "needs-key" in warnings[0] and "FAKE_KEY" in warnings[0]
    with pytest.raises(ValueError, match="FAKE_KEY"):
        KnowledgeBase.from_config(cfg, strict_reranker=True)


# --- writer lock -----------------------------------------------------------------------------------------

def test_writes_refuse_while_another_process_holds_the_store(config, notes, embedder):
    from basic_kb.lock import StoreBusy, WriterLock
    kb = KnowledgeBase(embedder, config.store_dir, writer_lock_timeout=0.1)
    other = WriterLock(config.store_dir)                 # a second handle stands in for another process
    other.acquire()
    try:
        with pytest.raises(StoreBusy):
            kb.index(notes, **CH)
        with pytest.raises(StoreBusy):
            kb.reindex_paths(notes, [notes.directory / "coffee.md"], **CH)
        with pytest.raises(StoreBusy):
            kb.vacuum()
        assert not kb.store.exists()                     # nothing was written
    finally:
        other.release()
    assert not kb.index(notes, **CH).aborted             # free again
    assert kb.search([notes], ["coffee"], n=1)           # reads never take the lock


def test_long_lived_holder_can_still_write_from_any_thread(indexed_kb, notes):
    """A server takes the lock once at startup and serves index() from handler threads."""
    import threading
    indexed_kb.writer_lock.acquire()
    try:
        out = []
        t = threading.Thread(target=lambda: out.append(indexed_kb.index(notes, **CH)))
        t.start()
        t.join(timeout=10)
        assert out and not out[0].aborted
        assert indexed_kb.writer_lock.held
    finally:
        indexed_kb.writer_lock.release()
    assert indexed_kb.writer_lock.is_free()


# --- helpers -------------------------------------------------------------------------------------------

def test_chunk_ids_are_content_derived_and_disambiguated():
    class C:
        def __init__(self, t): self.text = t
    ids = KnowledgeBase._chunk_ids("f.md", [C("a"), C("b"), C("a")])
    assert ids[0].startswith("f.md::") and ids[2] == ids[0] + "#1"
    assert ids == KnowledgeBase._chunk_ids("f.md", [C("a"), C("b"), C("a")])


def test_cores_to_threads():
    assert cores_to_threads(None) is None
    assert cores_to_threads(0) is None
    assert cores_to_threads(0.5) >= 1


def test_vacuum_returns_a_result(indexed_kb):
    res = indexed_kb.vacuum()
    assert res.vacuumed is True and res.live == 8 and res.deleted_since_vacuum == 0
    assert res.size_bytes > 0 and res.path.endswith("kb.sqlite3")
