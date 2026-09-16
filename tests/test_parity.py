"""The three surfaces expose the same operations. Drift here fails the suite."""
from __future__ import annotations

import inspect

import basic_kb
from basic_kb.cli import build_parser
from basic_kb.client import RemoteKnowledgeBase
from basic_kb.core import OPERATIONS, KnowledgeBase

# How each library operation is reached from the CLI. `keys`, `serve` and `watch` are
# process-level or administrative commands with no library operation behind them
# (ADR 0002 for keys) and are exempt by design.
CLI_FOR = {
    "search": ("search", None),
    "search_grouped": ("search", "--separate"),
    "index": ("index", None),
    "index_many": ("index", None),
    "preview": ("index", "--preview"),
    "status": ("status", None),
    "scan": ("scan", None),
    "info": ("info", None),
    "vacuum": ("vacuum", None),
}

# API route for each operation.
ROUTE_FOR = {
    "search": ("POST", "/search"), "search_grouped": ("POST", "/search"),
    "index": ("POST", "/index"), "index_many": ("POST", "/index"),
    "preview": ("POST", "/preview"), "status": ("GET", "/status"), "scan": ("GET", "/scan"),
    "info": ("GET", "/info"), "vacuum": ("POST", "/vacuum"),
}


def _subparsers(parser):
    for action in parser._actions:
        if isinstance(action, type(parser._subparsers._group_actions[0])):
            return action.choices
    raise AssertionError("no subparsers")


def test_every_operation_exists_on_local_and_remote():
    for op in OPERATIONS:
        assert callable(getattr(KnowledgeBase, op, None)), f"KnowledgeBase lacks {op}"
        assert callable(getattr(RemoteKnowledgeBase, op, None)), f"RemoteKnowledgeBase lacks {op}"


def test_remote_signatures_accept_the_local_call_shapes():
    """Every positional-or-keyword parameter of the local method (besides self and the
    callbacks a remote cannot stream) is accepted by the remote one, so a caller can
    swap the objects without changing the call."""
    for op in OPERATIONS:
        local = inspect.signature(getattr(KnowledgeBase, op)).parameters
        remote = inspect.signature(getattr(RemoteKnowledgeBase, op)).parameters
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in remote.values())
        for name in local:
            if name in ("self",) or accepts_kwargs:
                continue
            assert name in remote, f"RemoteKnowledgeBase.{op} lacks parameter {name!r}"


def test_every_operation_has_a_route(kb, config):
    from basic_kb.server import create_app, ServerState
    import threading, time
    app = create_app(ServerState(kb=kb, config=config, nonce="", started_at=time.time(), auth=False,
                                 keys=None, local_key=None, index_lock=threading.Lock()))
    routes = {(m, r.path) for r in app.routes for m in getattr(r, "methods", [])}
    for op in OPERATIONS:
        assert ROUTE_FOR[op] in routes, f"no route for {op}: {ROUTE_FOR[op]}"
    assert ("GET", "/health") in routes


def test_every_operation_has_a_cli_command():
    commands = _subparsers(build_parser())
    for op in OPERATIONS:
        cmd, flag = CLI_FOR[op]
        assert cmd in commands, f"no CLI command {cmd!r} for {op}"
        if flag:
            assert any(flag in a.option_strings for a in commands[cmd]._actions), f"{cmd} lacks {flag}"
    for exempt in ("keys", "serve", "watch"):
        assert exempt in commands


def test_module_entry_points_exist():
    assert callable(basic_kb.open) and callable(basic_kb.connect)
    assert "open" in basic_kb.__all__ and "connect" in basic_kb.__all__
