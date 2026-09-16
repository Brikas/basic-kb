"""served.json and the attach decision, including a server killed with SIGKILL."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from basic_kb.attach import ServedInfo, attach, read_served, remove_served, served_path, write_served
from basic_kb.client import RemoteKnowledgeBase
from basic_kb.lock import WriterLock

from .conftest import CONFIG_YAML

INFO = ServedInfo(url="http://127.0.0.1:1", nonce="abc", pid=1, hostname="h", model_id="m", version="0",
                  config_path="/c", started_at=1.0, auth=False)


def test_served_file_round_trip_and_corruption(tmp_path):
    assert read_served(tmp_path) is None
    write_served(tmp_path, INFO)
    assert read_served(tmp_path) == INFO
    assert not served_path(tmp_path).with_name("served.json.tmp").exists()   # atomic replace left no temp
    served_path(tmp_path).write_text("{torn", encoding="utf-8")
    assert read_served(tmp_path) is None and not served_path(tmp_path).exists()  # garbage is removed
    remove_served(tmp_path)                                                    # idempotent


# --- the decision procedure against a live server ---------------------------------------------------------

def notes_of(config, **kw):
    notes: list[str] = []
    result = attach(config, on_note=notes.append, **kw)
    return result, notes


def test_attaches_when_the_server_for_this_config_is_alive(served, config):
    remote, notes = notes_of(config)
    assert isinstance(remote, RemoteKnowledgeBase) and remote.url == served.info.url
    assert notes == []                                    # the common case is silent
    assert remote.info().total_chunks == 8


def test_no_attach_flag_and_env(served, config, monkeypatch):
    assert attach(config, no_attach=True) is None
    monkeypatch.setenv("BASIC_KB_NO_ATTACH", "1")
    assert attach(config) is None
    monkeypatch.setenv("BASIC_KB_NO_ATTACH", "0")
    assert attach(config) is not None


def test_attach_url_bypasses_local_state(served, config, tmp_path):
    remote = attach(config, attach_url=served.info.url)
    assert remote.url == served.info.url
    from basic_kb.client import RemoteError
    with pytest.raises(RemoteError):
        attach(config, attach_url="http://127.0.0.1:9")


def test_stale_file_with_free_lock_is_removed(config):
    write_served(config.store_dir, INFO)
    remote, notes = notes_of(config)
    assert remote is None and not served_path(config.store_dir).exists()
    assert notes and "stale" in notes[0]


def test_lock_held_without_a_file_means_a_plain_writer(config):
    holder = WriterLock(config.store_dir)
    holder.acquire()
    try:
        remote, notes = notes_of(config)
        assert remote is None and "another process is writing" in notes[0]
    finally:
        holder.release()


def test_file_from_another_machine_is_ignored(served, config):
    info = read_served(config.store_dir)
    write_served(config.store_dir, ServedInfo(**{**info.__dict__, "hostname": "some-other-box"}))
    remote, notes = notes_of(config)
    assert remote is None and "another machine" in notes[0]
    assert served_path(config.store_dir).exists()          # left for its owner


def test_nonce_mismatch_means_another_process_owns_the_port(served, config):
    info = read_served(config.store_dir)
    write_served(config.store_dir, ServedInfo(**{**info.__dict__, "nonce": "different"}))
    remote, notes = notes_of(config)
    assert remote is None and "not the basic-kb server" in notes[0]
    assert served_path(config.store_dir).exists()


def test_unreachable_url_while_lock_held_runs_locally(served, config):
    info = read_served(config.store_dir)
    write_served(config.store_dir, ServedInfo(**{**info.__dict__, "url": "http://127.0.0.1:9"}))
    remote, notes = notes_of(config)
    assert remote is None and "did not answer" in notes[0]
    assert served_path(config.store_dir).exists()


def test_other_config_or_model_runs_locally(served, config, instance):
    from dataclasses import replace
    other = replace(config, path=instance / "other.yaml")
    remote, notes = notes_of(other)
    assert remote is None and "started from" in notes[0]
    remote, notes = notes_of(config, model_override="different-model")
    assert remote is None and "--model" in notes[0]
    assert attach(config, model_override="fake-bow-64") is not None   # same model: fine


def test_auth_on_uses_the_local_key(served_auth, config):
    remote, notes = notes_of(config)
    assert remote is not None and remote.api_key == served_auth.local_key
    assert remote.health()["auth"] is True


# --- the crash: a real `basic-kb serve` subprocess killed with SIGKILL ------------------------------------

def test_sigkilled_server_leaves_nothing_that_misleads(tmp_path, monkeypatch):
    """No cleanup code runs in the killed process. The lock still frees (kernel), so the
    next CLI run treats served.json as stale, deletes it, and proceeds locally."""
    inst = tmp_path / "inst"
    (inst / "data" / "notes").mkdir(parents=True)
    (inst / "data" / "meetings").mkdir(parents=True)
    (inst / "basic-kb.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "basic_kb", "serve", "--config", str(inst / "basic-kb.yaml"), "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    served = served_path(inst / ".basic-kb")
    try:
        deadline = time.monotonic() + 30
        while not served.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"server never wrote served.json; exit={proc.poll()} stderr={proc.stderr.read()}")
            time.sleep(0.05)
        info = read_served(inst / ".basic-kb")
        assert info.pid == proc.pid and not WriterLock(inst / ".basic-kb").is_free()
        # The server answers and echoes its nonce.
        assert RemoteKnowledgeBase(info.url).health(timeout=5)["nonce"] == info.nonce
    finally:
        proc.kill()
        proc.wait(timeout=10)

    assert served.exists()                                # the crash left the file behind
    from basic_kb.config import load_config
    config = load_config(inst / "basic-kb.yaml")
    notes: list[str] = []
    assert attach(config, on_note=notes.append) is None
    assert not served.exists() and "stale" in notes[0]
    assert WriterLock(inst / ".basic-kb").is_free()
