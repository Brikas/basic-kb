"""JSON-safe conversion of the result dataclasses, shared by the CLI, the API and the client.

`to_jsonable` includes every `@property` of a dataclass alongside its fields, so derived
values (`embedded`, `stale`, `approx_tokens`) travel with the data. `from_dict` ignores
those extra keys and rebuilds the dataclass from its fields, so a value that went through
JSON comes back as the same type it left as. Nested dataclasses and lists of them are
handled from the type hints; no second model layer is needed.
"""
from __future__ import annotations

import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def to_jsonable(x: Any) -> Any:
    """Recursively convert dataclasses, lists, tuples, dicts and Paths into plain JSON values.
    Anything else passes through untouched (json.dumps decides whether it is acceptable)."""
    if is_dataclass(x) and not isinstance(x, type):
        out = {f.name: to_jsonable(getattr(x, f.name)) for f in fields(x)}
        # Derived values live as properties; a JSON reader wants them too. Walking the MRO
        # picks up properties defined on a base class as well.
        for klass in type(x).__mro__:
            for name, attr in vars(klass).items():
                if isinstance(attr, property) and name not in out:
                    out[name] = to_jsonable(getattr(x, name))
        return out
    if isinstance(x, (list, tuple)):
        return [to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, Path):
        return str(x)
    return x


def from_dict(cls: type[T], data: dict) -> T:
    """Rebuild dataclass `cls` from a dict produced by `to_jsonable`.

    Keys that are not fields (the properties) are ignored. A missing field with no
    default raises ValueError naming it, so a schema drift between server and client
    fails loudly instead of producing a half-filled object.
    """
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ValueError(f"{cls.__name__}: missing field {f.name!r} in {sorted(data)}")
            continue
        kwargs[f.name] = _coerce(hints.get(f.name), data[f.name])
    return cls(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    """Turn a JSON value back into the type the hint names: nested dataclasses, lists of
    them, Paths. Plain scalars pass through."""
    if value is None or hint is None:
        return value
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:      # Optional[X] / X | None
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value) if len(inner) == 1 else value
    if origin in (list, tuple) and args:
        return [_coerce(args[0], v) for v in value]
    if origin is dict:
        return dict(value)
    if isinstance(hint, type) and is_dataclass(hint) and isinstance(value, dict):
        return from_dict(hint, value)
    if hint is Path:
        return Path(value)
    return value
