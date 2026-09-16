"""The FastAPI app over a fake-embedder KnowledgeBase, through the in-process TestClient."""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from basic_kb.keys import ApiKeyStore
from basic_kb.server import ServerState, create_app, is_loopback

from .conftest import write_tree


def make_client(kb, config, *, auth=False, local_key=None) -> TestClient:
    state = ServerState(kb=kb, config=config, nonce="n0nce", started_at=time.time(), auth=auth,
                        keys=ApiKeyStore(config.store_dir) if auth else None, local_key=local_key,
                        index_lock=threading.Lock())
    return TestClient(create_app(state), raise_server_exceptions=False)


@pytest.fixture
def client(indexed_kb, config):
    return make_client(indexed_kb, config)


def test_is_loopback():
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("10.0.0.5") and not is_loopback("example.com")


# --- reads ---------------------------------------------------------------------------------------------

def test_health(client):
    body = client.get("/health").json()
    assert body["ok"] and body["nonce"] == "n0nce" and body["model_id"] == "fake-bow-64"
    assert body["sources"] == ["notes", "meetings"] and body["auth"] is False and body["pid"]


def test_info_status_scan(client):
    inf = client.get("/info").json()
    assert inf["name"] == "test-instance" and inf["total_chunks"] == 8
    st = client.get("/status", params={"source": "notes"}).json()
    assert len(st) == 1 and st[0]["chunks"] == 6 and st[0]["approx_tokens"] == st[0]["chars"] // 4
    sc = client.get("/scan").json()
    assert [s["source_id"] for s in sc] == ["notes", "meetings"] and all(s["stale"] == 0 for s in sc)


def test_unknown_source_is_400_with_lists(client):
    r = client.get("/status", params={"source": "nope"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "UnknownSource" and body["requested"] == ["nope"] and body["known"] == ["notes", "meetings"]


def test_search_fused_and_separate_with_notices(client):
    r = client.post("/search", json={"queries": ["burr grinder gooseneck kettle"], "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "fused" and 1 <= len(body["hits"]) <= 3 and "grinder" in body["hits"][0]["doc"]
    assert body["hits"][0]["sort_key"] == body["hits"][0]["score"]
    assert body["notices"] == []                    # freshness is disabled in the fixture config
    r = client.post("/search", json={"queries": ["coffee grinder", "tax receipts"], "separate": True, "n": 2,
                                     "sources": ["notes"]})
    body = r.json()
    assert body["mode"] == "separate" and [g["query"] for g in body["groups"]] == ["coffee grinder", "tax receipts"]
    assert all(len(g["hits"]) <= 2 for g in body["groups"])


def test_search_content_type_and_validation(client):
    r = client.post("/search", json={"queries": ["anything"], "content_type": "admin", "n": 10})
    assert r.json()["hits"] and all(h["metadata"]["content_type"] == "admin" for h in r.json()["hits"])
    r = client.post("/search", json={"queries": []})
    assert r.status_code == 400 and r.json()["error"] == "ValueError"
    r = client.post("/search", json={"nope": 1})
    assert r.status_code == 422                     # pydantic: missing `queries`


def test_search_without_index_is_404(kb, config, fake_provider):
    r = make_client(kb, config).post("/search", json={"queries": ["x"]})
    assert r.status_code == 404 and r.json()["error"] == "IndexNotFound"


# --- writes --------------------------------------------------------------------------------------------

def test_index_runs_and_returns_results(client, instance):
    (instance / "data" / "notes" / "new.md").write_text("# New\n\nA note added after the first index run.\n", encoding="utf-8")
    r = client.post("/index", json={"sources": "notes"})
    assert r.status_code == 200
    (res,) = r.json()
    assert res["source_id"] == "notes" and res["added"] == 1 and res["unchanged"] == 4 and res["aborted"] is False


def test_index_limit_budget_and_chunk_params(client):
    r = client.post("/index", json={"force": True, "limit": 1})
    assert [x["source_id"] for x in r.json()] == ["notes"] and r.json()[0]["limited_to"] == 1


def test_index_mass_change_is_409_with_numbers(indexed_kb, config, instance):
    write_tree(instance / "data" / "many", {f"n{i}.md": f"# N{i}\n\nEnough words for a chunk in note {i}.\n" for i in range(6)})
    cfg = instance / "basic-kb.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "  - id: many\n    type: markdown\n    path: data/many\n    chunker: recursive\n",
                   encoding="utf-8")
    from basic_kb.config import load_config
    config = load_config(cfg)
    client = make_client(indexed_kb, config)
    assert client.post("/index", json={"sources": "many"}).status_code == 200
    for f in (instance / "data" / "many").glob("*.md"):
        f.write_text(f.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    r = client.post("/index", json={"sources": "many"})
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "MassChangeRefused" and body["changed"] == 6 and body["base"] == 6
    r = client.post("/index", json={"sources": "many", "yes": True})
    assert r.status_code == 200 and r.json()[0]["updated"] == 6


def test_concurrent_index_is_409_busy(indexed_kb, config, monkeypatch):
    gate = threading.Event()
    real = indexed_kb.index_many

    def slow(*a, **k):
        gate.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(indexed_kb, "index_many", slow)
    client = make_client(indexed_kb, config)
    first: list = []
    t = threading.Thread(target=lambda: first.append(client.post("/index", json={"sources": "notes"})))
    t.start()
    time.sleep(0.2)
    second = client.post("/index", json={"sources": "notes"})
    gate.set()
    t.join(timeout=10)
    assert second.status_code == 409 and second.json()["error"] == "Busy"
    assert first[0].status_code == 200


def test_model_switch_is_refused_then_accepted(indexed_kb, config, fake_provider):
    from basic_kb.core import KnowledgeBase
    from .fakes import FakeEmbedder
    other = KnowledgeBase(FakeEmbedder(dim=16, model_id="other"), config.store_dir)
    client = make_client(other, config)
    r = client.post("/index", json={"sources": "notes"})
    assert r.status_code == 409 and r.json()["error"] == "StoreError" and "Model switch" in r.json()["message"]
    r = client.post("/index", json={"sources": "notes", "switch_model": True})
    assert r.status_code == 200 and [x["source_id"] for x in r.json()] == ["notes", "meetings"]


def test_vacuum_and_preview(client):
    v = client.post("/vacuum").json()
    assert v["vacuumed"] is True and v["live"] == 8
    p = client.post("/preview", json={"source": "notes", "limit": 1}).json()
    assert p[0]["rel_path"] == "coffee.md" and len(p[0]["chunks"]) == 3 and p[0]["skipped"] is False
    assert client.post("/preview", json={"source": "nope"}).status_code == 400


# --- authentication (ADR 0002) ---------------------------------------------------------------------------

def test_auth_off_is_open(client):
    assert client.get("/health").status_code == 200


def test_auth_on_requires_a_valid_key_everywhere(indexed_kb, config):
    store = ApiKeyStore(config.store_dir)
    rec, plain = store.create("autotemple")
    client = make_client(indexed_kb, config, auth=True, local_key="bkb_localkey")
    assert client.get("/health").status_code == 401
    assert client.get("/health").headers["WWW-Authenticate"] == "Bearer"
    assert client.get("/health").json()["error"] == "Unauthorized"
    assert client.post("/search", json={"queries": ["x"]}).status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer bkb_wrong"}).status_code == 401
    assert client.get("/health", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.get("/health", headers={"Authorization": f"Bearer {plain}"}).status_code == 200
    assert client.get("/health", headers={"Authorization": "bearer bkb_localkey"}).status_code == 200
    store.revoke(rec.id)                            # picked up without a restart
    assert client.get("/health", headers={"Authorization": f"Bearer {plain}"}).status_code == 401
