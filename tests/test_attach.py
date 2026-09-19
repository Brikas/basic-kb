"""The attach decision: a configured or given URL, or a local run. No discovery."""
from __future__ import annotations

from dataclasses import replace

import pytest

from basic_kb.attach import attach, resolve_attach_key
from basic_kb.client import RemoteError, RemoteKnowledgeBase
from basic_kb.errors import BasicKBError, IncompatibleVersion


# --- no URL means local ---------------------------------------------------------------------

def test_no_attach_url_runs_locally(config):
    assert attach(config) is None


def test_no_attach_flag_and_env_win_over_a_configured_url(served, config, monkeypatch):
    cfg = replace(config, attach_url=served.url)
    assert attach(cfg) is not None

    assert attach(cfg, no_attach=True) is None
    monkeypatch.setenv("BASIC_KB_NO_ATTACH", "1")
    assert attach(cfg) is None
    monkeypatch.setenv("BASIC_KB_NO_ATTACH", "0")
    assert attach(cfg) is not None


# --- a URL is a promise: it never degrades to a local run ---------------------------------

def test_configured_url_attaches(served, config):
    remote = attach(replace(config, attach_url=served.url))
    assert isinstance(remote, RemoteKnowledgeBase) and remote.url == served.url
    assert remote.info().total_chunks == 8


def test_explicit_url_attaches_and_overrides_the_configured_one(served, config):
    notes: list[str] = []
    cfg = replace(config, attach_url="http://127.0.0.1:9")
    remote = attach(cfg, attach_url=served.url, on_note=notes.append)
    assert remote.url == served.url
    assert notes and "overrides the configured" in notes[0]


def test_unreachable_server_raises_rather_than_running_locally(config):
    cfg = replace(config, attach_url="http://127.0.0.1:9")
    with pytest.raises(RemoteError, match="cannot reach"):
        attach(cfg)
    with pytest.raises(RemoteError, match="cannot reach"):
        attach(config, attach_url="http://127.0.0.1:9")


def test_version_gap_raises_rather_than_running_locally(served, config, monkeypatch):
    import basic_kb.client as client_mod
    monkeypatch.setattr(client_mod, "__version__", "0.1.0")
    with pytest.raises(IncompatibleVersion):
        attach(replace(config, attach_url=served.url))


# --- the key ----------------------------------------------------------------------------------

def test_key_env_that_is_not_set_stops_the_run(served, config, monkeypatch):
    """Calling anyway would surface as a 401 and send you looking at the server."""
    monkeypatch.delenv("PKB_TEST_KEY", raising=False)
    cfg = replace(config, attach_url=served.url, attach_key_env="PKB_TEST_KEY")
    with pytest.raises(BasicKBError, match="PKB_TEST_KEY, but it is not set"):
        attach(cfg)


def test_key_comes_from_the_env_var_the_config_names(served_auth, config, monkeypatch):
    monkeypatch.setenv("PKB_TEST_KEY", served_auth.local_key)
    monkeypatch.delenv("BASIC_KB_API_KEY", raising=False)
    cfg = replace(config, attach_url=served_auth.url, attach_key_env="PKB_TEST_KEY")
    assert attach(cfg).health()["auth"] is True


def test_key_precedence(config, monkeypatch):
    monkeypatch.setenv("PKB_TEST_KEY", "from-config-env")
    monkeypatch.setenv("BASIC_KB_API_KEY", "from-default-env")
    cfg = replace(config, attach_key_env="PKB_TEST_KEY")
    assert resolve_attach_key(cfg, "explicit") == "explicit"
    assert resolve_attach_key(cfg) == "from-config-env"
    assert resolve_attach_key(config) == "from-default-env"


def test_bad_key_is_refused_by_the_server(served_auth, config):
    cfg = replace(config, attach_url=served_auth.url)
    with pytest.raises(RemoteError, match="unauthorized"):
        attach(cfg, api_key="bkb_wrong")


def test_key_file_is_read_when_the_client_does_not_sit_next_to_the_store(served_auth, config, tmp_path, monkeypatch):
    """A thin client config has no store of its own, so the server's local key has to be
    named explicitly rather than found beside it."""
    monkeypatch.delenv("BASIC_KB_API_KEY", raising=False)
    key_file = tmp_path / "local.key"
    key_file.write_text(served_auth.local_key + "\n", encoding="utf-8")
    cfg = replace(config, attach_url=served_auth.url, attach_key_file=key_file,
                  store_dir=tmp_path / "no-store-here")
    assert attach(cfg).health()["auth"] is True


def test_missing_or_empty_key_file_stops_the_run(config, tmp_path, monkeypatch):
    monkeypatch.delenv("BASIC_KB_API_KEY", raising=False)
    cfg = replace(config, attach_url="http://127.0.0.1:9", attach_key_file=tmp_path / "absent.key")
    with pytest.raises(BasicKBError, match="probably not running"):
        attach(cfg)

    empty = tmp_path / "empty.key"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(BasicKBError, match="is empty"):
        attach(replace(cfg, attach_key_file=empty))
