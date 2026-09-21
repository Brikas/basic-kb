"""CLI: every command driven through main(), JSON output asserted structurally."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import basic_kb.cli as cli
from basic_kb.embedders import EMBEDDER_PROVIDERS
from basic_kb.sources import MarkdownSource

from .conftest import write_tree
from .fakes import FakeEmbedder


@pytest.fixture
def fake_cli(monkeypatch):
    """Register a `fake` embedding provider and make it the default, so the CLI builds the
    fake embedder through the same registry a real plugin would use. Records each
    (config, threads) the factory saw."""
    seen: list = []

    def build(config, threads=None):
        seen.append((config, threads))
        return FakeEmbedder()

    monkeypatch.setitem(EMBEDDER_PROVIDERS, "fake", build)
    monkeypatch.setitem(EMBEDDER_PROVIDERS, "local", build)     # the fixture config names no provider
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    return seen


@pytest.fixture
def run(fake_cli, instance, capsys):
    """run(*args) -> (stdout, stderr); pass --config automatically."""
    cfg = str(instance / "basic-kb.yaml")

    def _run(*args: str, expect_exit: int | None = None):
        argv = list(args)
        if argv and argv[0] not in ("--inspect", "help") and "--config" not in argv:
            argv += ["--config", cfg]
        if expect_exit is None:
            cli.main(argv)
        else:
            with pytest.raises(SystemExit) as e:
                cli.main(argv)
            assert e.value.code == expect_exit
        out = capsys.readouterr()
        return out.out, out.err

    return _run


def loads(out: str):
    return json.loads(out)


# --- help / config resolution --------------------------------------------------------------------------

def test_bare_invocation_prints_help(run):
    out, _ = run()
    assert "Common usage" in out
    out, _ = run("help")
    assert "Common usage" in out


def test_no_config_found_exits_1(fake_cli, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["status"])
    assert e.value.code == 1
    assert "No config found" in capsys.readouterr().err


def test_config_discovered_from_cwd(fake_cli, instance, monkeypatch, capsys):
    monkeypatch.chdir(instance)
    cli.main(["info", "--json"])
    assert loads(capsys.readouterr().out)["name"] == "test-instance"


def test_inspect_prints_freshness_template(run, instance, monkeypatch):
    monkeypatch.chdir(instance)              # --inspect takes no --config; it uses discovery
    out, _ = run("--inspect")
    assert "Freshness nudge" in out and "placeholders" in out


# --- index -------------------------------------------------------------------------------------------------

def test_index_json(run):
    out, _ = run("index", "--json")
    results = loads(out)
    assert [r["source_id"] for r in results] == ["notes", "meetings"]
    notes = results[0]
    assert (notes["added"], notes["empty"], notes["files_on_disk"]) == (3, 1, 4)
    assert notes["embedded"] == 3                 # property included
    assert notes["aborted"] is False and notes["errors"] == []


def test_index_human_output_has_progress(run):
    out, _ = run("index", "--source", "notes")
    assert "Indexing Test Notes" in out and "Done [notes]" in out


def test_index_unknown_source_exits_1(run):
    _, err = run("index", "--source", "nope", expect_exit=1)
    assert "Unknown source 'nope'" in err and "notes, meetings" in err


def test_index_limit_is_a_whole_run_budget(run):
    out, _ = run("index", "--limit", "1", "--json")
    results = loads(out)
    assert len(results) == 1                      # budget spent on the first source
    assert results[0]["limited_to"] == 1
    out, _ = run("index", "--limit", "1", "--limit-per-source", "--force", "--json")
    assert len(loads(out)) == 2


def test_index_preview_writes_file_without_store(run, instance):
    target = instance / "preview.txt"
    out, _ = run("index", "--preview", "--source", "notes", "--out", str(target))
    assert f"Preview written to: {target}" in out
    text = target.read_text(encoding="utf-8")
    assert "FILE: coffee.md" in text and "Total: 4 file(s), 6 chunks" in text   # stub.md: parsed, 0 chunks
    assert not (instance / ".basic-kb" / "kb.sqlite3").exists()


def test_model_flag_overrides_config_for_the_run(run, fake_cli):
    run("info", "--model", "other-model")
    assert fake_cli[-1][0].embedding_model == "other-model"
    run("info")
    assert fake_cli[-1][0].embedding_model == "fake-bow-64"


def test_throttle_flag_maps_to_threads(run, fake_cli):
    run("index", "--source", "notes", "--cores-fraction", "0.5")
    assert fake_cli[-1][1] is not None and fake_cli[-1][1] >= 1


# --- mass-change guard through the CLI --------------------------------------------------------------------

@pytest.fixture
def many_instance(instance):
    write_tree(instance / "data" / "many", {
        f"n{i}.md": f"# Note {i}\n\nThis is note number {i} with enough words to make a chunk.\n" for i in range(6)})
    cfg = instance / "basic-kb.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "  - id: many\n    type: markdown\n    path: data/many\n    chunker: recursive\n",
                   encoding="utf-8")
    return instance


def rewrite(dirpath: Path):
    for f in MarkdownSource("m", dirpath).get_files():
        f.write_text(f.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")


def test_guard_abort_is_exit_code_1_unattended(run, many_instance):
    run("index", "--source", "many")
    rewrite(many_instance / "data" / "many")
    _, err = run("index", "--source", "many", expect_exit=1)
    assert "Refusing to re-embed unattended" in err


def test_guard_yes_accepts(run, many_instance):
    run("index", "--source", "many")
    rewrite(many_instance / "data" / "many")
    out, _ = run("index", "--source", "many", "--yes", "--json")
    assert loads(out)[0]["updated"] == 6


def test_guard_can_be_skipped_for_one_run(run, many_instance):
    run("index", "--source", "many")
    rewrite(many_instance / "data" / "many")
    out, _ = run("index", "--source", "many", "--no-reindex-guard", "--json")
    assert loads(out)[0]["aborted"] is False


# --- search ---------------------------------------------------------------------------------------------------

def test_search_requires_a_query(run):
    run("index", "--json")
    _, err = run("search", expect_exit=1)
    assert "at least one query" in err


def test_search_json_fused(run):
    run("index", "--json")
    out, _ = run("search", "burr grinder gooseneck kettle", "--json", "--n", "3")
    doc = loads(out)
    assert doc["mode"] == "fused" and doc["queries"] == ["burr grinder gooseneck kettle"]
    assert 1 <= len(doc["hits"]) <= 3
    hit = doc["hits"][0]
    assert set(hit) >= {"doc", "metadata", "score", "rerank_score", "sort_key"}
    assert "grinder" in hit["doc"]


def test_search_offset_pages(run):
    run("index", "--json")
    whole = loads(run("search", "coffee burr grinder", "--json", "--n", "4")[0])["hits"]
    page, _ = run("search", "coffee burr grinder", "--json", "--n", "2", "--offset", "2")
    assert [h["doc"] for h in loads(page)["hits"]] == [h["doc"] for h in whole[2:4]]


def test_search_json_separate(run):
    run("index", "--json")
    out, _ = run("search", "coffee grinder", "tax receipts", "--separate", "--json", "--n", "2")
    doc = loads(out)
    assert doc["mode"] == "separate"
    assert [g["query"] for g in doc["groups"]] == ["coffee grinder", "tax receipts"]
    assert all(len(g["hits"]) <= 2 for g in doc["groups"])


def test_search_human_output(run):
    run("index", "--json")
    out, _ = run("search", "burr grinder", "--source", "notes", "--n", "2")
    assert out.startswith("Top ")
    assert "Coffee brewing" in out and "[notes · hobby]" in out
    assert "score=" in out


def test_search_max_chars_truncates(run):
    run("index", "--json")
    out, _ = run("search", "burr grinder", "--source", "notes", "--n", "1", "--max-chars", "10")
    assert "more chars]" in out


def test_search_source_list_exits_0(run):
    out, _ = run("search", "--source", "list", expect_exit=0)
    assert "notes" in out and "meetings" in out and "All sources combined" in out


def test_search_without_index_is_an_error_not_empty(run):
    _, err = run("search", "anything", expect_exit=1)
    assert "Error: No index" in err


def test_strict_rerank_without_reranker_exits_1(run):
    run("index", "--json")
    _, err = run("search", "x", "--rerank", expect_exit=1)
    assert "--rerank set but no reranker chosen" in err


def test_no_rerank_flag_is_accepted(run):
    run("index", "--json")
    out, _ = run("search", "coffee", "--no-rerank", "--json")
    assert loads(out)["hits"]


# --- status / scan / info / vacuum ------------------------------------------------------------------------------

def test_status_json_includes_properties(run):
    run("index", "--json")
    out, _ = run("status", "--json")
    st = loads(out)
    assert [s["source_id"] for s in st] == ["notes", "meetings"]
    n = st[0]
    assert n["chunks"] == 6 and n["docs_with_chunks"] == 3 and n["files_on_disk"] == 4
    assert n["pending"] == 0 and n["stale"] == 0 and n["approx_tokens"] == n["chars"] // 4
    assert n["content_types"] == {"hobby": 4, "admin": 2}


def test_status_human_output(run):
    run("index", "--json")
    out, _ = run("status", "--source", "notes")
    assert "Source  : Test Notes  (notes)" in out
    assert "State   : up to date. Last index: just now" in out
    assert "hobby: 4 chunks" in out


def test_status_before_index_exits_1(run):
    _, err = run("status", expect_exit=1)
    assert "No index" in err


def test_scan_json_and_human(run, instance):
    out, _ = run("scan", "--json", "--source", "notes")
    assert loads(out)[0]["tracked"] is False
    run("index", "--json")
    (instance / "data" / "notes" / "coffee.md").write_text("# C\n\nchanged body with enough words in it.\n", encoding="utf-8")
    out, _ = run("scan", "--source", "notes")
    assert "changed=1" in out and "1 stale" in out and "Re-index with" in out


def test_info_json(run):
    run("index", "--json")
    out, _ = run("info", "--json")
    inf = loads(out)
    assert inf["name"] == "test-instance" and inf["total_files"] == 6 and inf["total_chunks"] == 8
    assert inf["sources"][0]["description"] == "Markdown notes with frontmatter."
    assert inf["sources"][0]["chunks_per_file"] == 1.5


def test_info_works_before_any_index(run):
    out, _ = run("info")
    assert "[NOT INDEXED]" in out


def test_vacuum_json(run):
    run("index", "--json")
    out, _ = run("vacuum", "--json")
    doc = loads(out)
    assert doc["vacuumed"] is True and doc["deleted_since_vacuum"] == 0 and doc["live"] == 8


# --- freshness nudge ----------------------------------------------------------------------------------------------

def test_freshness_nudge_appears_counts_and_clears(fake_cli, instance, capsys):
    cfg = instance / "basic-kb.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("freshness:\n  enabled: false",
                                                             "freshness:\n  enabled: true\n  stale_after_days: 0\n  remind_every_days: 0"),
                   encoding="utf-8")
    c = ["--config", str(cfg)]
    cli.main(["index", "--json"] + c)
    capsys.readouterr()
    cli.main(["search", "coffee", "--source", "notes", "--json"] + c)
    assert "[basic-kb]" not in capsys.readouterr().err                    # up to date: silent
    (instance / "data" / "notes" / "coffee.md").write_text("# C\n\nchanged body with enough words in it.\n", encoding="utf-8")
    cli.main(["search", "coffee", "--source", "notes"] + c)
    err = capsys.readouterr().err
    assert "[basic-kb] Source 'notes' looks stale" in err and "reminder 1" in err
    state = json.loads((instance / ".basic-kb" / "freshness_state.json").read_text())
    assert state["notes"]["nudges"] == 1
    cli.main(["status", "--source", "notes"] + c)
    assert "nudges 1" in capsys.readouterr().out
    cli.main(["index", "--source", "notes", "--json"] + c)
    assert "notes" not in json.loads((instance / ".basic-kb" / "freshness_state.json").read_text())


def test_json_search_nudges_on_stderr(fake_cli, instance, capsys):
    """A --json caller is usually an agent, the reader the nudge exists for; stderr keeps the document clean."""
    cfg = instance / "basic-kb.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("freshness:\n  enabled: false",
                                                             "freshness:\n  enabled: true\n  stale_after_days: 0\n  remind_every_days: 0"),
                   encoding="utf-8")
    c = ["--config", str(cfg)]
    cli.main(["index", "--json"] + c)
    (instance / "data" / "notes" / "coffee.md").write_text("# C\n\nchanged body with enough words in it.\n", encoding="utf-8")
    capsys.readouterr()
    cli.main(["search", "coffee", "--source", "notes", "--json"] + c)
    out = capsys.readouterr()
    assert "[basic-kb] Source 'notes' looks stale" in out.err
    json.loads(out.out)                                      # stdout is still one clean document


# --- keys (ADR 0002) ---------------------------------------------------------------------------------------------

def test_keys_create_list_revoke(run, instance):
    out, err = run("keys", "create", "--name", "laptop")
    key = out.strip()
    assert key.startswith("bkb_") and "only time it is shown" in err
    out, _ = run("keys", "list")
    assert "laptop" in out and key not in out and "active" in out
    out, _ = run("keys", "list", "--json")
    (rec,) = loads(out)
    assert rec["name"] == "laptop" and rec["active"] is True and "sha256" in rec
    out, _ = run("keys", "revoke", rec["id"])
    assert "Revoked" in out
    out, _ = run("keys", "list")
    assert "revoked" in out
    _, err = run("keys", "revoke", "laptop", expect_exit=1)
    assert "no active key" in err


def test_keys_create_json(run):
    out, _ = run("keys", "create", "--name", "x", "--json")
    doc = loads(out)
    assert doc["key"].startswith("bkb_") and doc["prefix"] == doc["key"][:10]


def test_keys_list_when_empty(run):
    out, _ = run("keys", "list")
    assert "No API keys" in out


# --- serve ------------------------------------------------------------------------------------------------------------

def test_serve_refuses_non_loopback_without_auth(run):
    _, err = run("serve", "--host", "0.0.0.0", expect_exit=1)
    assert "refusing to serve on 0.0.0.0 without authentication" in err


def test_serve_refuses_when_another_writer_holds_the_store(run, instance, monkeypatch):
    from basic_kb.lock import WriterLock
    holder = WriterLock(instance / ".basic-kb")
    holder.acquire()
    try:
        _, err = run("serve", "--port", "0", expect_exit=1)
        assert "another process is writing" in err
    finally:
        holder.release()


# --- attach ------------------------------------------------------------------------------------------------------------

@pytest.fixture
def live(run, fake_cli, instance):
    """Index locally, start a server on a free port, then point the config at it — which is
    how a machine with a served instance is actually set up now."""
    from basic_kb.config import load_config
    from basic_kb.server import KBServer
    run("index", "--json")
    cfg_path = instance / "basic-kb.yaml"
    server = KBServer(load_config(cfg_path), port=0)
    server.start()
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + f"\nattach_cli:\n  url: {server.url}\n",
                        encoding="utf-8")
    try:
        yield server
    finally:
        server.stop()


def test_cli_attaches_and_builds_no_embedder(run, fake_cli, live, capsys):
    before = len(fake_cli)
    out, err = run("search", "burr grinder gooseneck kettle", "--json", "--n", "3")
    doc = loads(out)
    assert doc["mode"] == "fused" and "grinder" in doc["hits"][0]["doc"]
    assert len(fake_cli) == before                      # no local embedder was constructed
    assert "[basic-kb]" not in err                      # the common case is silent
    out, _ = run("status", "--json")
    assert loads(out)[0]["chunks"] == 6
    out, _ = run("info")
    assert "test-instance" in out
    out, _ = run("scan", "--json")
    assert all(s["stale"] == 0 for s in loads(out))
    out, _ = run("vacuum", "--json")
    assert loads(out)["vacuumed"] is True
    assert len(fake_cli) == before


def test_cli_index_goes_through_the_server(run, fake_cli, live, instance):
    (instance / "data" / "notes" / "new.md").write_text("# New\n\nAdded while served, long enough to chunk.\n", encoding="utf-8")
    before = len(fake_cli)
    out, _ = run("index", "--source", "notes")
    assert "Indexing on the served instance" in out and "Done [notes]" in out
    assert len(fake_cli) == before
    out, _ = run("index", "--source", "notes", "--json")
    assert loads(out)[0]["unchanged"] == 5


def test_cli_no_attach_runs_locally_and_writes_refuse(run, fake_cli, live):
    before = len(fake_cli)
    out, _ = run("search", "coffee", "--json", "--no-attach")
    assert loads(out)["hits"] and len(fake_cli) == before + 1      # a local embedder was built
    _, err = run("index", "--source", "notes", "--no-attach", expect_exit=1)
    assert "another process is writing the store" in err


def test_cli_attach_url(run, fake_cli, live):
    before = len(fake_cli)
    out, _ = run("info", "--json", "--attach", live.url)
    assert loads(out)["name"] == "test-instance" and len(fake_cli) == before
    _, err = run("info", "--attach", "http://127.0.0.1:9", expect_exit=1)
    assert "cannot reach" in err


def test_cli_attach_with_auth_uses_local_key(run, fake_cli, instance):
    """On the box, the server's local.key means no key has to be typed."""
    from basic_kb.config import load_config
    from basic_kb.server import KBServer
    run("index", "--json")
    cfg_path = instance / "basic-kb.yaml"
    server = KBServer(load_config(cfg_path), port=0, auth=True)
    server.start()
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + f"\nattach_cli:\n  url: {server.url}\n",
                        encoding="utf-8")
    try:
        before = len(fake_cli)
        out, _ = run("status", "--json")
        assert loads(out)[0]["chunks"] == 6 and len(fake_cli) == before
        _, err = run("status", "--attach", server.url, "--api-key", "bkb_wrong", expect_exit=1)
        assert "unauthorized" in err
    finally:
        server.stop()


# --- an attached run needs no local instance (the server owns the config) ------------------

def test_attach_needs_no_local_config_at_all(fake_cli, live, tmp_path, monkeypatch, capsys):
    """A remote node should not have to duplicate the box's source list to query it."""
    monkeypatch.chdir(tmp_path)                       # nowhere near a basic-kb.yaml
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    cli.main(["status", "--json", "--attach", live.url])
    st = json.loads(capsys.readouterr().out)
    assert [s["source_id"] for s in st] == ["notes", "meetings"]


def test_attach_resolves_source_ids_on_the_server(fake_cli, live, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    cli.main(["search", "coffee", "--json", "--source", "notes", "--attach", live.url])
    assert json.loads(capsys.readouterr().out)["hits"]

    with pytest.raises(SystemExit) as e:
        cli.main(["status", "--source", "nope", "--attach", live.url])
    assert e.value.code == 1
    assert "Unknown source 'nope'" in capsys.readouterr().err


def test_attach_source_list_describes_the_remote(fake_cli, live, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    with pytest.raises(SystemExit) as e:
        cli.main(["search", "--source", "list", "--attach", live.url])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "notes" in out and "Markdown notes with frontmatter." in out


def test_no_config_error_mentions_attach(fake_cli, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["status"])
    assert "--attach URL" in capsys.readouterr().err


# --- response shaping: what a machine reader gets ------------------------------------------

def test_json_search_strips_bookkeeping_metadata_by_default(run):
    run("index", "--json")
    out, _ = run("search", "burr grinder", "--source", "notes", "--json", "--n", "1")
    meta = loads(out)["hits"][0]["metadata"]
    assert "rel_path" in meta and "title" in meta and "source" in meta
    for dead in ("content_hash", "position", "chunk_index", "file"):
        assert dead not in meta, dead


def test_detailed_keeps_everything(run):
    run("index", "--json")
    out, _ = run("search", "burr grinder", "--source", "notes", "--json", "--n", "1", "--detailed")
    meta = loads(out)["hits"][0]["metadata"]
    assert {"content_hash", "position", "file"} <= set(meta)


def test_json_search_honours_max_chars(run):
    run("index", "--json")
    out, _ = run("search", "burr grinder", "--source", "notes", "--json", "--n", "1", "--max-chars", "40")
    assert len(loads(out)["hits"][0]["doc"]) == 40
    out, _ = run("search", "burr grinder", "--source", "notes", "--json", "--n", "1")
    assert len(loads(out)["hits"][0]["doc"]) > 40          # full text by default


def test_status_says_where_it_read_from(run, fake_cli, live, instance):
    """A status from a served instance and one from the local store can disagree, so the
    output has to say which it was."""
    out, _ = run("status", "--source", "notes")
    assert f"Reading : {live.url}  (attached)" in out

    out, _ = run("status", "--source", "notes", "--no-attach")
    assert "(local store)" in out and "kb.sqlite3" in out


def test_serve_ignores_attach_cli_and_never_calls_itself(fake_cli, instance, capsys):
    """An instance config may carry `attach_cli:` so a CLI run from its folder talks to the
    server. `serve` must read that as 'not for me' rather than attaching to itself."""
    from basic_kb.cli import LOCAL_ONLY_COMMANDS
    assert {"serve", "watch"} <= set(LOCAL_ONLY_COMMANDS)

    cfg = instance / "basic-kb.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "\nattach_cli:\n  url: http://127.0.0.1:9\n",
                   encoding="utf-8")
    # Port 9 refuses connections, so attaching would fail loudly. Binding wide without auth
    # is the first thing serve checks, which proves it got past attach resolution entirely.
    with pytest.raises(SystemExit):
        cli.main(["serve", "--host", "0.0.0.0", "--config", str(cfg)])
    assert "refusing to serve on 0.0.0.0" in capsys.readouterr().err
