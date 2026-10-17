"""Load an object named in config by its import path, `package.module:Name` (ADR 0005).

Built-in components keep their short names in their registries. A config value containing
a colon is an import path instead: the module is imported with the config file's folder on
`sys.path`, so a source type can be a loose file next to the instance config, with no
install step and no reinstall after an edit.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional


def is_import_path(value: str) -> bool:
    """True for `package.module:Name`, the form that names code outside the engine."""
    return ":" in str(value)


def load_import_path(spec: str, base_dir: Optional[Path] = None, what: str = "object") -> object:
    """Import `package.module:Name` and return `Name`.

    `base_dir` (the config's folder) is put on `sys.path` first, so a module sitting next to
    the config imports without being installed. Every failure raises ValueError naming the
    path and `what` was being loaded — a broken plugin must surface at config load, never as
    a later AttributeError somewhere in the engine.
    """
    module_name, _, attr = str(spec).partition(":")
    if not module_name or not attr:
        raise ValueError(f"{what} {spec!r}: an import path has the form 'package.module:Name'.")
    if base_dir is not None:
        anchor = str(Path(base_dir).resolve())
        if anchor not in sys.path:
            sys.path.insert(0, anchor)
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise ValueError(
            f"{what} {spec!r}: cannot import module {module_name!r} "
            f"(searched {base_dir} and sys.path): {type(e).__name__}: {e}") from e
    obj = module
    for part in attr.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"{what} {spec!r}: module {module_name!r} has no attribute {attr!r}.")
        obj = getattr(obj, part)
    return obj
