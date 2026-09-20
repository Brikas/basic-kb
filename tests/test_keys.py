"""API key store and local key (ADR 0002)."""
from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from basic_kb.keys import ApiKeyStore, KeyError_, LocalKey, generate_key


def test_generate_key_shape():
    k = generate_key()
    assert k.startswith("bkb_") and len(k) == 4 + 32 and k != generate_key()


def test_create_list_verify_revoke(tmp_path):
    store = ApiKeyStore(tmp_path)
    rec, plain = store.create("laptop")
    assert rec.name == "laptop" and rec.active and rec.prefix == plain[:10]
    assert plain not in store.path.read_text()                # only the hash is stored
    assert store.verify(plain) == rec
    assert store.verify("bkb_wrong") is None and store.verify("") is None
    assert [k.id for k in store.list()] == [rec.id]

    gone = store.revoke(rec.id)
    assert gone.id == rec.id and not gone.active
    assert store.verify(plain) is None
    assert store.active() == [] and len(store.list()) == 1


def test_several_keys_are_independent(tmp_path):
    store = ApiKeyStore(tmp_path)
    _, a = store.create("a")
    rb, b = store.create("b")
    store.revoke("a")
    assert store.verify(a) is None and store.verify(b) == rb


def test_names_must_be_unique_among_active_keys(tmp_path):
    store = ApiKeyStore(tmp_path)
    store.create("x")
    with pytest.raises(KeyError_, match="already exists"):
        store.create("x")
    store.revoke("x")
    store.create("x")                                          # a revoked name can be reused
    with pytest.raises(KeyError_, match="needs a name"):
        store.create("   ")


def test_revoke_by_name_or_id_and_errors(tmp_path):
    store = ApiKeyStore(tmp_path)
    r1, _ = store.create("dup")
    store.revoke("dup")
    r2, _ = store.create("dup")
    assert store.revoke("dup").id == r2.id                     # only one active match
    with pytest.raises(KeyError_, match="no active key"):
        store.revoke("dup")
    with pytest.raises(KeyError_, match="no active key"):
        store.revoke("nope")


def test_revocation_is_picked_up_by_another_store_object(tmp_path):
    """The server holds one ApiKeyStore for its lifetime; `keys revoke` writes through another."""
    server_side = ApiKeyStore(tmp_path)
    admin = ApiKeyStore(tmp_path)
    _, plain = admin.create("k")
    assert server_side.verify(plain) is not None
    admin.revoke("k")
    assert server_side.verify(plain) is None


def test_corrupt_file_is_an_error_not_an_open_door(tmp_path):
    store = ApiKeyStore(tmp_path)
    store.path.write_text("{oops", encoding="utf-8")
    with pytest.raises(KeyError_, match="not valid JSON"):
        store.verify("bkb_x")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_files_are_owner_only(tmp_path):
    store = ApiKeyStore(tmp_path)
    store.create("k")
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    lk = LocalKey(tmp_path)
    lk.write()
    assert stat.S_IMODE(os.stat(lk.path).st_mode) == 0o600


def test_local_key_round_trip(tmp_path):
    lk = LocalKey(tmp_path)
    assert lk.read() is None
    k = lk.write()
    assert lk.read() == k and k.startswith("bkb_")
    assert lk.write() != k                                      # a new key per start
    lk.remove()
    assert lk.read() is None
    lk.remove()                                                 # idempotent
