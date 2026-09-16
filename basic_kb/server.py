"""The served API: a FastAPI app over one KnowledgeBase, and the process around it.

`create_app` builds the routes; `KBServer` owns the process-level pieces: the writer
lock held for the server's lifetime (lock.py), `served.json` and the local key for
same-machine attach (attach.py, keys.py), optional bearer-key authentication
(ADR 0002), and an optional in-process watcher so one process owns the loaded models
and every write to the store.

Routes mirror KnowledgeBase one to one (the parity test enforces it):

    GET  /health           identity: name, version, model, nonce, pid, sources, auth
    GET  /info             KnowledgeBase.info
    GET  /status?source=   KnowledgeBase.status
    GET  /scan?source=     KnowledgeBase.scan per source
    POST /search           search / search_grouped, plus freshness `notices`
    POST /index            index_many (synchronous; a second run while one is active is 409)
    POST /vacuum           vacuum
    POST /preview          preview

Responses are the library's dataclasses through `serialize.to_jsonable`, properties
included, so a CLI that attaches renders exactly what a local run would. Errors are
`{error: <exception class>, message, ...}` with the class name the client rebuilds:
UnknownSource 400, IndexNotFound 404, MassChangeRefused/StoreBusy/Busy/StoreError 409,
EmbeddingError/QueryFailed 502, Unauthorized 401.

`fastapi` and `uvicorn` come from the `serve` extra; importing this module without them
raises a BasicKBError that says so.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

from .attach import ServedInfo, remove_served, write_served
from .config import Config
from .core import DEFAULT_N, KnowledgeBase
from .errors import (
    BasicKBError, EmbeddingError, IndexNotFound, MassChangeRefused, QueryFailed, StoreError, UnknownSource,
)
from .freshness import FreshnessTracker
from .keys import ApiKeyStore, LocalKey
from .lock import StoreBusy
from .serialize import to_jsonable
from .sources import resolve_sources
from .version import __version__
from .watcher import Watcher, WatchEvent, resolve_settings

logger = logging.getLogger("basic_kb")

try:
    from fastapi import FastAPI, Depends, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as _e:   # pragma: no cover - exercised only without the extra installed
    raise BasicKBError("`basic-kb serve` needs the serve extra: pip install 'basic-kb[serve]'") from _e


class Busy(BasicKBError):
    """An index run is already in progress on this server."""


class Unauthorized(BasicKBError):
    """No valid API key on a server with authentication on."""


def is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class ServerState:
    kb: KnowledgeBase
    config: Config
    nonce: str
    started_at: float
    auth: bool
    keys: Optional[ApiKeyStore]
    local_key: Optional[str]
    index_lock: threading.Lock


# --- request bodies ------------------------------------------------------------------------------

Selector = Union[str, list[str]]


class SearchRequest(BaseModel):
    queries: list[str]
    sources: Selector = "all"
    separate: bool = False
    n: Optional[int] = None
    content_type: Optional[str] = None
    rerank_candidates: Optional[int] = None
    cand_multiplier: Optional[int] = None
    cand_min: Optional[int] = None
    cand_max: Optional[int] = None
    strict_rerank: bool = False


class IndexRequest(BaseModel):
    sources: Selector = "all"
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    min_chunk: Optional[int] = None
    force: bool = False
    limit: Optional[int] = None
    limit_per_source: bool = False
    switch_model: bool = False
    pause_ms: Optional[int] = None
    pause_every: Optional[int] = None
    guard: Optional[bool] = None
    guard_threshold: Optional[float] = None
    yes: bool = False


class PreviewRequest(BaseModel):
    source: str
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    min_chunk: Optional[int] = None
    limit: Optional[int] = None
    file_name: Optional[str] = None


# --- the app -----------------------------------------------------------------------------------------

def _state(request: Request) -> ServerState:
    return request.app.state.kb


def require_key(request: Request) -> None:
    """Global dependency: with authentication on, every route wants a bearer key that is
    either this start's local key or an active key in api-keys.json."""
    st = _state(request)
    if not st.auth:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if token:
        if st.local_key and hmac.compare_digest(token, st.local_key):
            return
        if st.keys is not None and st.keys.verify(token) is not None:
            return
    client = request.client.host if request.client else "?"
    logger.warning("auth failed: client=%s key_prefix=%s path=%s", client, token[:10] or "-", request.url.path)
    raise Unauthorized("a valid API key is required: Authorization: Bearer <key>")


def _chunk_params(cfg: Config, chunk_size, overlap, min_chunk) -> dict:
    return dict(chunk_size=chunk_size or cfg.chunk_size,
                overlap=cfg.overlap if overlap is None else overlap,
                min_chunk=min_chunk or cfg.min_chunk)


def create_app(state: ServerState) -> "FastAPI":
    app = FastAPI(title=f"basic-kb: {state.config.name}", version=__version__,
                  dependencies=[Depends(require_key)])
    app.state.kb = state

    @app.exception_handler(BasicKBError)
    def _kb_error(request: Request, exc: BasicKBError):
        status, extra = 500, {}
        headers = None
        if isinstance(exc, UnknownSource):
            status, extra = 400, {"requested": exc.requested, "known": exc.known}
        elif isinstance(exc, IndexNotFound):
            status = 404
        elif isinstance(exc, MassChangeRefused):
            status = 409
            extra = {"source_id": exc.source_id, "changed": exc.changed, "deleted": exc.deleted,
                     "base": exc.base, "fraction": exc.fraction, "threshold": exc.threshold}
        elif isinstance(exc, (StoreBusy, Busy, StoreError)):
            status = 409
        elif isinstance(exc, (EmbeddingError, QueryFailed)):
            status = 502
        elif isinstance(exc, Unauthorized):
            status, headers = 401, {"WWW-Authenticate": "Bearer"}
        if status == 500:
            logger.exception("server error on %s", request.url.path)
        return JSONResponse({"error": type(exc).__name__, "message": str(exc), **extra},
                            status_code=status, headers=headers)

    @app.exception_handler(ValueError)
    def _value_error(request: Request, exc: ValueError):
        return JSONResponse({"error": "ValueError", "message": str(exc)}, status_code=400)

    @app.get("/health")
    def health(request: Request):
        st = _state(request)
        return {"ok": True, "name": st.config.name, "version": __version__, "model_id": st.kb.embedder.model_id,
                "nonce": st.nonce, "pid": os.getpid(), "started_at": st.started_at,
                "sources": [s["id"] for s in st.config.sources], "auth": st.auth}

    @app.get("/info")
    def info(request: Request, source: str = "all"):
        st = _state(request)
        return to_jsonable(st.kb.info(resolve_sources(st.config, source), name=st.config.name))

    @app.get("/status")
    def status(request: Request, source: str = "all"):
        st = _state(request)
        return to_jsonable(st.kb.status(resolve_sources(st.config, source)))

    @app.get("/scan")
    def scan(request: Request, source: str = "all"):
        st = _state(request)
        return to_jsonable([st.kb.scan(s) for s in resolve_sources(st.config, source)])

    @app.post("/search")
    def search(req: SearchRequest, request: Request):
        st = _state(request)
        cfg = st.config
        if not req.queries:
            raise ValueError("at least one query is required")
        content_type = req.content_type or cfg.search_content_type
        sources = resolve_sources(cfg, req.sources, content_type)
        kwargs = dict(
            sources=sources, queries=req.queries, n=req.n or cfg.search_n or DEFAULT_N,
            content_type_filter=content_type, rerank_candidates=req.rerank_candidates,
            cand_multiplier=req.cand_multiplier or cfg.cand_multiplier,
            cand_min=req.cand_min or cfg.cand_min, cand_max=req.cand_max or cfg.cand_max,
            strict_rerank=req.strict_rerank,
        )
        if req.separate:
            groups = st.kb.search_grouped(**kwargs)
            body = {"mode": "separate", "groups": [{"query": q, "hits": to_jsonable(h)} for q, h in groups]}
        else:
            body = {"mode": "fused", "queries": req.queries, "hits": to_jsonable(st.kb.search(**kwargs))}
        body["notices"] = FreshnessTracker.from_config(cfg).evaluate(st.kb, sources)
        return body

    @app.post("/index")
    def index(req: IndexRequest, request: Request):
        st = _state(request)
        cfg = st.config
        if not st.index_lock.acquire(blocking=False):
            raise Busy("an index run is already in progress on this server; retry when it finishes")
        try:
            sources = resolve_sources(cfg, "all" if req.switch_model else req.sources)

            def refuse(detail: MassChangeRefused) -> bool:
                raise detail            # the API never prompts; the caller gets a 409 with the numbers

            results = st.kb.index_many(
                sources, **_chunk_params(cfg, req.chunk_size, req.overlap, req.min_chunk),
                force=req.force, limit=req.limit, limit_per_source=req.limit_per_source,
                switch_model=req.switch_model,
                pause_ms=cfg.throttle_pause_ms if req.pause_ms is None else req.pause_ms,
                pause_every=cfg.throttle_pause_every if req.pause_every is None else req.pause_every,
                guard=cfg.reindex_guard if req.guard is None else req.guard,
                guard_threshold=cfg.reindex_guard_threshold if req.guard_threshold is None else req.guard_threshold,
                assume_yes=req.yes, on_confirm=refuse,
            )
        finally:
            st.index_lock.release()
        return to_jsonable(results)

    @app.post("/vacuum")
    def vacuum(request: Request):
        return to_jsonable(_state(request).kb.vacuum())

    @app.post("/preview")
    def preview(req: PreviewRequest, request: Request):
        st = _state(request)
        (source,) = resolve_sources(st.config, [req.source])
        files = st.kb.preview(source, **_chunk_params(st.config, req.chunk_size, req.overlap, req.min_chunk),
                              limit=req.limit, file_name=req.file_name)
        return to_jsonable(files)

    return app


# --- the process -------------------------------------------------------------------------------------

class KBServer:
    """Serve one instance. `serve_forever()` for a foreground command (uvicorn handles
    SIGINT/SIGTERM on the main thread); `start()`/`stop()` for tests and embedders."""

    def __init__(
        self,
        config: Config,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        auth: Optional[bool] = None,
        watch: Optional[bool] = None,
        allow_unauthenticated: bool = False,
        model: Optional[str] = None,
        threads: Optional[int] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        min_chunk: Optional[int] = None,
        debounce: Optional[int] = None,
        on_event: Optional[Callable[[str], None]] = None,
        on_watch_event: Optional[Callable[[WatchEvent], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        import uvicorn  # noqa: F401  (part of the serve extra; fail here, before any state changes)

        self.config = config
        self.host = host or config.serve_host
        self.port = config.serve_port if port is None else port
        self.auth = config.serve_auth if auth is None else auth
        self.watch = config.serve_watch if watch is None else watch
        if not is_loopback(self.host) and not self.auth and not allow_unauthenticated:
            raise ValueError(
                f"refusing to serve on {self.host} without authentication. Turn it on with `serve.auth: true` "
                f"in the config or --auth (then `basic-kb keys create --name NAME`), or pass "
                f"--allow-unauthenticated when something in front of basic-kb handles access control.")
        self.chunk = _chunk_params(config, chunk_size, overlap, min_chunk)
        self.debounce = debounce
        self.on_event = on_event
        self.on_watch_event = on_watch_event
        self.kb = KnowledgeBase.from_config(config, model=model, threads=threads, on_warning=on_warning)
        self.info: Optional[ServedInfo] = None
        self.local_key: Optional[str] = None
        self._uv = None
        self._sock = None
        self._thread: Optional[threading.Thread] = None
        self._watcher: Optional[Watcher] = None
        self._watcher_thread: Optional[threading.Thread] = None

    def _emit(self, msg: str) -> None:
        logger.info("serve: %s", msg)
        if self.on_event is not None:
            self.on_event(msg)

    def _prepare(self) -> ServedInfo:
        import uvicorn

        cfg = self.config
        store_dir = cfg.store_dir
        # For the server's lifetime: a second server, an `index` in another shell or a
        # watcher now fail with StoreBusy instead of racing this process.
        self.kb.writer_lock.acquire(timeout=0)
        try:
            all_sources = resolve_sources(cfg, "all")
            # Refuse to serve a store built with another model; every search would be
            # refused anyway, and a watcher would idle forever. The message says what to run.
            self.kb.prepare_model_switch(all_sources, accept=False)

            keys = ApiKeyStore(store_dir) if self.auth else None
            if self.auth:
                self.local_key = LocalKey(store_dir).write()
                if not keys.active():
                    self._emit("authentication is on and no API keys exist yet: only same-machine attach "
                               "works until you run `basic-kb keys create --name NAME`")
            state = ServerState(kb=self.kb, config=cfg, nonce=secrets.token_hex(8), started_at=time.time(),
                                auth=self.auth, keys=keys, local_key=self.local_key, index_lock=threading.Lock())
            app = create_app(state)

            uvconfig = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning",
                                      log_config=None, access_log=False)
            self._sock = uvconfig.bind_socket()
            bound_host, bound_port = self._sock.getsockname()[:2]
            # served.json is read by a CLI on this machine: a wildcard bind is reachable via loopback.
            url_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
            if ":" in url_host:
                url_host = f"[{url_host}]"
            self._uv = uvicorn.Server(uvconfig)

            if self.watch:
                raw_by_id = {s["id"]: s for s in cfg.sources}
                watched = [(s, resolve_settings(raw_by_id.get(s.source_id, {}), self.debounce)) for s in all_sources]
                self._watcher = Watcher(self.kb, watched, **self.chunk, guard=cfg.reindex_guard,
                                        guard_threshold=cfg.reindex_guard_threshold,
                                        on_event=self.on_watch_event or (lambda ev: None))

            self.info = ServedInfo(url=f"http://{url_host}:{bound_port}", nonce=state.nonce, pid=os.getpid(),
                                   hostname=socket.gethostname(), model_id=self.kb.embedder.model_id,
                                   version=__version__, config_path=str(cfg.path), started_at=state.started_at,
                                   auth=self.auth)
            write_served(store_dir, self.info)
            return self.info
        except BaseException:
            self._cleanup()
            raise

    def _start_watcher(self) -> None:
        """The startup reconcile can take a while; it runs beside the HTTP server, which is
        already answering, rather than delaying it."""
        if self._watcher is None:
            return
        self._watcher_thread = threading.Thread(target=self._watcher.start, name="basic-kb-watch", daemon=True)
        self._watcher_thread.start()

    def start(self, timeout: float = 15.0) -> ServedInfo:
        """Run in a background thread; returns once the server answers."""
        info = self._prepare()
        self._thread = threading.Thread(target=self._uv.run, kwargs={"sockets": [self._sock]},
                                        name="basic-kb-serve", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._uv.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                self.stop()
                raise BasicKBError("the server did not start within the timeout")
            time.sleep(0.02)
        self._start_watcher()
        return info

    def serve_forever(self) -> None:
        """Foreground: uvicorn owns the main thread and turns SIGINT/SIGTERM into a clean exit."""
        info = self._prepare()
        self._emit(f"serving '{self.config.name}' at {info.url} (pid {info.pid}, auth "
                   f"{'on' if self.auth else 'off'}, watch {'on' if self.watch else 'off'}); Ctrl-C to stop")
        # The watcher starts once uvicorn is listening; uvicorn calls startup handlers itself,
        # so hang the watcher start off a tiny delayed thread instead.
        threading.Timer(0.5, self._start_watcher).start()
        try:
            self._uv.run(sockets=[self._sock])
        finally:
            self._cleanup()
            self._emit("stopped")

    def stop(self, timeout: float = 15.0) -> None:
        if self._uv is not None:
            self._uv.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            if self._watcher_thread is not None:
                self._watcher_thread.join(timeout=30)
            self._watcher = None
        remove_served(self.config.store_dir)
        LocalKey(self.config.store_dir).remove()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        while self.kb.writer_lock.held:
            self.kb.writer_lock.release()
