"""RemoteKnowledgeBase: the KnowledgeBase surface over HTTP.

Used by the CLI when it attaches to a served instance, and by module users via
`basic_kb.connect(url, api_key)`. Method names, arguments and return types match
`KnowledgeBase`, so a caller works against either without knowing which it holds.
Sources may be passed as DataSourceBase objects or as ids; only ids travel.

Errors come back as the same typed exceptions the local library raises, rebuilt from
the server's `{error, message, ...}` payload, so `except IndexNotFound:` works on both
sides. Transport failures and unmapped answers raise RemoteError.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Union

import requests

from .errors import (
    BasicKBError, EmbeddingError, IndexNotFound, MassChangeRefused, QueryFailed, StoreError, UnknownSource,
)
from .models import (
    IndexResult, InstanceInfo, PreviewFile, ScanResult, SearchResult, SourceStatus, VacuumResult,
)
from .serialize import from_dict

API_KEY_ENV = "BASIC_KB_API_KEY"


class RemoteError(BasicKBError):
    """The server could not be reached, or answered with something the client cannot map."""


_ERROR_TYPES: dict[str, type[BasicKBError]] = {
    cls.__name__: cls for cls in (BasicKBError, StoreError, EmbeddingError, IndexNotFound, QueryFailed)
}


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """The key to send: an explicit value, else `BASIC_KB_API_KEY` (which the config's
    `env_file` may have loaded), else nothing."""
    return explicit or os.environ.get(API_KEY_ENV) or None


def _ids(sources: Union[None, str, list]) -> Optional[str]:
    """Comma-separated ids for the wire; None means the server's default (all)."""
    if sources is None or sources == "all":
        return None
    if isinstance(sources, str):
        return sources
    return ",".join(getattr(s, "source_id", str(s)) for s in sources)


class RemoteKnowledgeBase:
    def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = 120.0,
                 session: Optional[requests.Session] = None) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = session or requests.Session()
        self.last_notices: list[str] = []     # freshness nudges from the last search

    # --- transport -----------------------------------------------------------------------

    def _request(self, method: str, path: str, *, json: Any = None, params: Optional[dict] = None,
                 timeout: Optional[float] = None) -> Any:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = self._session.request(method, self.url + path, json=json, params=params, headers=headers,
                                      timeout=self.timeout if timeout is None else timeout)
        except requests.RequestException as e:
            raise RemoteError(f"cannot reach the basic-kb server at {self.url}: {e}") from e
        if r.status_code >= 400:
            raise self._error(r)
        return r.json()

    def _error(self, r: requests.Response) -> BasicKBError:
        try:
            payload = r.json() if r.content else {}
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        name = payload.get("error", "")
        message = payload.get("message") or r.text[:300] or f"HTTP {r.status_code}"
        if name == "MassChangeRefused":
            try:
                return MassChangeRefused(payload["source_id"], payload["changed"], payload["deleted"],
                                         payload["base"], payload["fraction"], payload["threshold"])
            except KeyError:
                return StoreError(message)
        if name == "UnknownSource":
            return UnknownSource(payload.get("requested", []), payload.get("known", []))
        if name == "StoreBusy":
            from .lock import StoreBusy
            return StoreBusy(message)
        cls = _ERROR_TYPES.get(name)
        if cls is not None:
            return cls(message)
        if r.status_code == 401:
            return RemoteError(f"unauthorized by {self.url}: {message}. Pass an API key with --api-key, "
                               f"{API_KEY_ENV}, or the config's env_file.")
        return RemoteError(f"server error {r.status_code} from {self.url}: {message}")

    # --- the KnowledgeBase surface ----------------------------------------------------------

    def health(self, timeout: Optional[float] = None) -> dict:
        return self._request("GET", "/health", timeout=timeout)

    def info(self, sources=None, name: str = "") -> InstanceInfo:
        params = {"source": _ids(sources)} if _ids(sources) else None
        return from_dict(InstanceInfo, self._request("GET", "/info", params=params))

    def status(self, sources=None) -> list[SourceStatus]:
        params = {"source": _ids(sources)} if _ids(sources) else None
        return [from_dict(SourceStatus, d) for d in self._request("GET", "/status", params=params)]

    def scan(self, source) -> ScanResult:
        data = self._request("GET", "/scan", params={"source": _ids([source])})
        return from_dict(ScanResult, data[0])

    def _search(self, separate: bool, sources, queries: list[str], n: Optional[int], content_type_filter,
                rerank_candidates, cand_multiplier, cand_min, cand_max, strict_rerank) -> dict:
        body = {"queries": list(queries), "sources": _ids(sources) or "all", "separate": separate,
                "n": n, "content_type": content_type_filter, "rerank_candidates": rerank_candidates,
                "cand_multiplier": cand_multiplier, "cand_min": cand_min, "cand_max": cand_max,
                "strict_rerank": strict_rerank}
        data = self._request("POST", "/search", json=body)
        self.last_notices = list(data.get("notices", []))
        return data

    def search(self, sources, queries: list[str], n: Optional[int] = None, content_type_filter=None,
               rerank_candidates=None, cand_multiplier=None, cand_min=None, cand_max=None,
               strict_rerank: bool = False, timing: bool = False) -> list[SearchResult]:
        data = self._search(False, sources, queries, n, content_type_filter, rerank_candidates,
                            cand_multiplier, cand_min, cand_max, strict_rerank)
        return [from_dict(SearchResult, h) for h in data["hits"]]

    def search_grouped(self, sources, queries: list[str], n: Optional[int] = None, content_type_filter=None,
                       rerank_candidates=None, cand_multiplier=None, cand_min=None, cand_max=None,
                       strict_rerank: bool = False, timing: bool = False) -> list[tuple[str, list[SearchResult]]]:
        data = self._search(True, sources, queries, n, content_type_filter, rerank_candidates,
                            cand_multiplier, cand_min, cand_max, strict_rerank)
        return [(g["query"], [from_dict(SearchResult, h) for h in g["hits"]]) for g in data["groups"]]

    def index_many(self, sources, chunk_size=None, overlap=None, min_chunk=None, force: bool = False,
                   limit: Optional[int] = None, limit_per_source: bool = False, switch_model: bool = False,
                   pause_ms: int = 0, pause_every: int = 50, guard: bool = True, guard_threshold: float = 0.9,
                   assume_yes: bool = False, on_progress: Optional[Callable[[str], None]] = None,
                   on_confirm: Optional[Callable[[MassChangeRefused], bool]] = None) -> list[IndexResult]:
        """One index run on the server. Progress is not streamed; `on_progress` gets a line
        before the request and one summary per source after. The server never prompts: a
        mass change comes back as MassChangeRefused, and if `on_confirm` accepts it the
        run is repeated with `assume_yes`."""
        body = {"sources": _ids(sources) or "all", "chunk_size": chunk_size, "overlap": overlap,
                "min_chunk": min_chunk, "force": force, "limit": limit, "limit_per_source": limit_per_source,
                "switch_model": switch_model, "pause_ms": pause_ms, "pause_every": pause_every,
                "guard": guard, "guard_threshold": guard_threshold, "yes": assume_yes}
        if on_progress is not None:
            on_progress(f"Indexing on the served instance at {self.url} (progress is reported when it finishes)…")
        try:
            data = self._request("POST", "/index", json=body, timeout=None)
        except MassChangeRefused as refused:
            if on_confirm is None or not on_confirm(refused):
                raise
            body["yes"] = True
            data = self._request("POST", "/index", json=body, timeout=None)
        results = [from_dict(IndexResult, d) for d in data]
        if on_progress is not None:
            for r in results:
                on_progress(f"Done [{r.source_id}]: new={r.added} updated={r.updated} unchanged={r.unchanged} "
                            f"empty={r.empty} pruned={r.pruned}; chunks {r.chunks_embedded} embedded, "
                            f"{r.chunks_reused} reused, {r.total_chunks} total"
                            + (f"  ABORTED: {r.abort_reason}" if r.aborted else ""))
        return results

    def index(self, source, **kwargs) -> IndexResult:
        return self.index_many([source], **kwargs)[0]

    def vacuum(self) -> VacuumResult:
        return from_dict(VacuumResult, self._request("POST", "/vacuum"))

    def preview(self, source, chunk_size=None, overlap=None, min_chunk=None, limit: Optional[int] = None,
                file_name: Optional[str] = None) -> list[PreviewFile]:
        body = {"source": _ids([source]), "chunk_size": chunk_size, "overlap": overlap, "min_chunk": min_chunk,
                "limit": limit, "file_name": file_name}
        return [from_dict(PreviewFile, d) for d in self._request("POST", "/preview", json=body)]
