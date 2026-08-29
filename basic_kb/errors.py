"""Typed errors.

The library raises these instead of printing a warning and continuing. An empty
result must always mean "nothing matched" — never "the store was missing", "the
model was wrong", or "the query blew up". A caller that cannot tell those apart
will serve a broken index as if it were an empty one.
"""
from __future__ import annotations


class BasicKBError(Exception):
    """Base class for everything this library raises deliberately."""


class StoreError(BasicKBError):
    """The store itself is unusable: extension missing, model/dimension mismatch, ..."""


class EmbeddingError(BasicKBError):
    """The embedding backend could not produce vectors: missing API key, provider error,
    wrong dimensionality. Raised rather than returning empties — an empty vector would
    silently poison the index."""


class IndexNotFound(BasicKBError):
    """No index exists for the requested source(s), so no query could run."""


class QueryFailed(BasicKBError):
    """A collection query failed. Distinct from a query that matched nothing."""


class MassChangeRefused(BasicKBError):
    """The mass-change guard tripped and no caller accepted the change.

    Carries the numbers so a caller can decide, prompt, or report. The library
    never prompts on its own — see `KnowledgeBase.index(on_confirm=...)`.
    """

    def __init__(self, source_id: str, changed: int, deleted: int, base: int,
                 fraction: float, threshold: float) -> None:
        self.source_id = source_id
        self.changed = changed
        self.deleted = deleted
        self.base = base
        self.fraction = fraction
        self.threshold = threshold
        super().__init__(
            f"Mass change on {source_id!r}: {changed} changed + {deleted} deleted "
            f"of {base} indexed files = {round(fraction * 100)}% "
            f"(guard threshold {round(threshold * 100)}%). Index left unchanged."
        )
