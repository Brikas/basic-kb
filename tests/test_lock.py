"""WriterLock: exclusivity across handles, threads and processes; release on abrupt death."""
from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from basic_kb.lock import StoreBusy, WriterLock


def test_acquire_release_and_probe(tmp_path):
    lock = WriterLock(tmp_path / "store")
    assert lock.is_free() and not lock.held
    with lock:
        assert lock.held and lock.path.exists()
        assert not lock.is_free()                       # the probe sees our own hold
    assert not lock.held and lock.is_free()


def test_second_handle_is_refused_with_store_busy(tmp_path):
    a = WriterLock(tmp_path, timeout=0.1)
    b = WriterLock(tmp_path, timeout=0.1)
    a.acquire()
    try:
        with pytest.raises(StoreBusy, match="another process is writing"):
            b.acquire()
    finally:
        a.release()
    b.acquire()
    b.release()


def test_reentrant_across_threads_within_one_process(tmp_path):
    lock = WriterLock(tmp_path)
    lock.acquire()
    seen = []

    def worker():
        with lock:                                      # same object, other thread: no wait
            seen.append(lock.held)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)
    assert seen == [True] and lock.held                 # still held by the outer acquire
    lock.release()
    assert not lock.held


HOLDER = """
import sys, time
from basic_kb.lock import WriterLock
lock = WriterLock(sys.argv[1])      # keep a reference: a dropped lock object releases on garbage collection
lock.acquire()
print("held", flush=True)
time.sleep(60)
"""


def test_lock_held_by_another_process_and_freed_when_it_dies(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        lock = WriterLock(tmp_path, timeout=0.2)
        assert not lock.is_free()
        with pytest.raises(StoreBusy):
            lock.acquire()
    finally:
        proc.kill()                                     # SIGKILL: no cleanup code runs in the holder
        proc.wait(timeout=10)
    lock = WriterLock(tmp_path, timeout=2)
    assert lock.is_free()
    with lock:
        assert lock.held
