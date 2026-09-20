"""RemoteKnowledgeBase against a live in-thread server: same results as the local library."""
from __future__ import annotations

import pytest

from basic_kb.client import RemoteError, RemoteKnowledgeBase
from basic_kb.errors import IndexNotFound, MassChangeRefused, UnknownSource
from basic_kb.keys import ApiKeyStore
from basic_kb.models import IndexResult, InstanceInfo, PreviewFile, ScanResult, SearchResult, SourceStatus, VacuumResult

from .conftest import write_tree

CH = dict(chunk_size=400, overlap=40, min_chunk=20)


@pytest.fixture
def remote(served) -> RemoteKnowledgeBase:
    return RemoteKnowledgeBase(served.url)


def test_health_and_info(remote, served):
    h = remote.health()
    assert h["name"] == "test-instance" and h["auth"] is False
    inf = remote.info()
    assert isinstance(inf, InstanceInfo) and inf.total_chunks == 8 and inf.sources[0].chunks_per_file == 1.5


def test_status_and_scan_accept_objects_or_ids(remote, sources, notes):
    st = remote.status(sources)
    assert [type(s) for s in st] == [SourceStatus, SourceStatus] and st[0].pending == 0
    assert remote.status("notes")[0].chunks == 6
    sc = remote.scan(notes)
    assert isinstance(sc, ScanResult) and sc.unchanged == 4


def test_search_matches_local(remote, indexed_kb, sources):
    local = indexed_kb.search(sources, ["burr grinder gooseneck kettle"], n=3)
    over_http = remote.search(sources, ["burr grinder gooseneck kettle"], n=3)
    assert all(isinstance(h, SearchResult) for h in over_http)
    assert [h.doc for h in over_http] == [h.doc for h in local]
    assert [round(h.score, 6) for h in over_http] == [round(h.score, 6) for h in local]
    assert remote.last_notices == []


def test_search_grouped(remote, notes):
    groups = remote.search_grouped([notes], ["coffee grinder", "tax receipts"], n=2)
    assert [q for q, _ in groups] == ["coffee grinder", "tax receipts"]
    assert all(isinstance(h, SearchResult) for _, hits in groups for h in hits)


def test_errors_arrive_as_the_same_types(remote, config, fake_provider, kb):
    with pytest.raises(UnknownSource) as e:
        remote.status("nope")
    assert e.value.requested == ["nope"] and e.value.known == ["notes", "meetings"]


def test_index_not_found_over_http(config, fake_provider, kb):
    from basic_kb.server import KBServer
    server = KBServer(config, port=0)         # fresh store: nothing indexed
    server.start()
    try:
        with pytest.raises(IndexNotFound):
            RemoteKnowledgeBase(server.url).search("all", ["x"])
    finally:
        server.stop()


def test_index_many_and_index(remote, instance, sources, notes):
    (instance / "data" / "notes" / "new.md").write_text("# New\n\nA note added after the first run, long enough.\n", encoding="utf-8")
    lines: list[str] = []
    results = remote.index_many([notes], on_progress=lines.append, **CH)
    assert [type(r) for r in results] == [IndexResult] and results[0].added == 1
    assert lines[0].startswith("Indexing on the served instance") and lines[1].startswith("Done [notes]")
    assert remote.index(notes, **CH).unchanged == 5


def test_index_mass_change_prompts_through_on_confirm(served, instance, config):
    write_tree(instance / "data" / "many", {f"n{i}.md": f"# N{i}\n\nEnough words for a chunk in note {i}.\n" for i in range(6)})
    cfg = instance / "basic-kb.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "  - id: many\n    type: markdown\n    path: data/many\n    chunker: recursive\n",
                   encoding="utf-8")
    # The running server still has the old config; restart on the same store with the new one.
    served.stop()
    from basic_kb.config import load_config
    from basic_kb.server import KBServer
    server = KBServer(load_config(cfg), port=0)
    server.start()
    try:
        remote = RemoteKnowledgeBase(server.url)
        remote.index_many("many", **CH)
        for f in (instance / "data" / "many").glob("*.md"):
            f.write_text(f.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
        with pytest.raises(MassChangeRefused) as e:
            remote.index_many("many", **CH)
        assert e.value.changed == 6 and e.value.fraction == 1.0
        asked: list = []
        results = remote.index_many("many", on_confirm=lambda d: asked.append(d) or True, **CH)
        assert len(asked) == 1 and results[0].updated == 6
    finally:
        server.stop()


def test_vacuum_and_preview(remote, notes):
    v = remote.vacuum()
    assert isinstance(v, VacuumResult) and v.vacuumed and v.live == 8
    files = remote.preview(notes, limit=1, **CH)
    assert isinstance(files[0], PreviewFile) and files[0].rel_path == "coffee.md" and len(files[0].chunks) == 3


def test_unreachable_server_is_a_remote_error():
    with pytest.raises(RemoteError, match="cannot reach"):
        RemoteKnowledgeBase("http://127.0.0.1:9", timeout=1).health()


def test_auth_over_http(served_auth, config):
    url = served_auth.url
    with pytest.raises(RemoteError, match="unauthorized"):
        RemoteKnowledgeBase(url).health()
    assert RemoteKnowledgeBase(url, api_key=served_auth.local_key).health()["auth"] is True
    store = ApiKeyStore(config.store_dir)
    rec, plain = store.create("laptop")
    assert RemoteKnowledgeBase(url, api_key=plain).info().total_chunks == 8
    store.revoke(rec.id)
    with pytest.raises(RemoteError, match="unauthorized"):
        RemoteKnowledgeBase(url, api_key=plain).health()


def test_connect_helper(served, monkeypatch):
    import basic_kb
    monkeypatch.setenv("BASIC_KB_API_KEY", "bkb_from_env")
    remote = basic_kb.connect(served.url)
    assert isinstance(remote, RemoteKnowledgeBase) and remote.api_key == "bkb_from_env"
    assert basic_kb.connect(served.url, api_key="explicit").api_key == "explicit"


# --- wire compatibility between a client and a served instance --------------------------

def test_health_advertises_the_minimum_client(remote):
    h = remote.health()
    assert h["min_client_version"] and h["version"]


def test_client_too_old_is_refused_loudly(served, monkeypatch):
    """A half-upgraded pair must fail once at connect, not at some later missing field."""
    import basic_kb.client as client_mod
    from basic_kb.errors import IncompatibleVersion

    monkeypatch.setattr(client_mod, "__version__", "0.1.0")
    with pytest.raises(IncompatibleVersion, match="needs a client of at least"):
        RemoteKnowledgeBase(served.url).health()


def test_server_too_old_is_refused_loudly(served, monkeypatch):
    import basic_kb.client as client_mod
    from basic_kb.errors import IncompatibleVersion

    monkeypatch.setattr(client_mod, "MIN_SERVER_VERSION", "99.0.0")
    with pytest.raises(IncompatibleVersion, match="this client needs at least"):
        RemoteKnowledgeBase(served.url).health()


