"""Instance configuration.

An *instance* is one config file describing a store directory, embedding model,
chunker defaults, and a list of sources. The config file's directory is the
anchor for every relative path it contains (store_dir, source paths, env_file).

Secrets never live in the config — only a *path* to a dotenv file (env_file),
or you set the variable in the environment / via --env-file. See README.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chunkers import DEFAULT_CHUNK_SIZE, DEFAULT_MIN_CHUNK, DEFAULT_OVERLAP
from .embedders import DEFAULT_MODEL


def load_env_file(path: Path) -> int:
    """Load KEY=VALUE lines from a dotenv file into os.environ via setdefault.

    setdefault means already-set environment variables win — so an explicit
    shell export always overrides the file. Returns the number of keys seen.
    """
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        count += 1
    return count


# Conventional config filename discovered by walking up from the working dir.
CONFIG_FILENAMES = ("basic-kb.yaml", "basic-kb.yml")

# Local-override suffix: basic-kb.local.yaml / basic-kb.local.yml sit alongside
# the main config and deep-merge on top of it. Gitignored; machine-specific paths go here.
_LOCAL_SUFFIXES = (".local.yaml", ".local.yml")
# Shown after a search when a source hasn't been re-indexed in a while and has
# drifted. Placeholders: {source} {new} {updated} {deleted} {unchanged} {stale}
# {total} {days}. Override via `freshness.message` in the config.
DEFAULT_FRESHNESS_MESSAGE = (
    "[basic-kb] Source '{source}' looks stale since the last index: "
    "{new} new, {updated} changed, {deleted} deleted file(s). "
    "Consider prompting the user if they want to re-index (basic-kb index --source {source})."
)


def find_config(start: Optional[Path] = None) -> Optional[Path]:
    """Locate an instance config the way git/npm find theirs.

    Precedence: $BASIC_KB_CONFIG, else walk up from `start` (default: cwd)
    looking for basic-kb.yaml. Returns None if nothing is found — the caller
    decides how to report that.
    """
    env = os.environ.get("BASIC_KB_CONFIG")
    if env:
        return Path(env).expanduser()
    start = (start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place). Lists replace entirely."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve(base_dir: Path, raw: str) -> Path:
    """Absolute paths pass through; relative paths anchor to the config's dir."""
    p = Path(raw).expanduser()
    return p if p.is_absolute() else base_dir / p


@dataclass
class Config:
    path: Path           # the config file itself
    base_dir: Path       # directory containing the config (anchor for rel paths)
    name: str
    store_dir: Path      # ChromaDB persistent store for this instance
    embedding_model: str
    chunk_size: int
    overlap: int
    min_chunk: int
    sources: list[dict]  # raw source entries; built via sources.build_source()
    reranker_type: str = "none"        # none | local | jina
    reranker_model: Optional[str] = None
    cand_multiplier: int = 3           # candidates to rerank = clamp(n*mult, min, max)
    cand_min: int = 50
    cand_max: int = 200
    timing: bool = False               # print per-phase search timings
    freshness_enabled: bool = True     # post-search staleness reminder
    freshness_every_days: int = 3      # how often to re-check a source
    freshness_message: str = DEFAULT_FRESHNESS_MESSAGE
    throttle_cores: Optional[float] = None   # fraction of CPU cores for indexing (None = all)
    throttle_priority: str = "normal"        # normal | low (OS process priority while indexing)
    throttle_pause_ms: int = 0               # sleep this many ms ...
    throttle_pause_every: int = 50           # ... every N embedded files
    log_file: Optional[Path] = None          # if set, write event log here (relative to config dir)
    log_level: str = "INFO"
    log_max_bytes: int = 10_000_000          # rotate at ~10 MB ...
    log_backup_count: int = 5                # ... keeping 5 old files (~50 MB of history)
    reindex_guard: bool = True               # confirm before re-embedding a mass-changed source
    reindex_guard_threshold: float = 0.9     # churn fraction (changed+deleted / indexed) that triggers it
    env_file: Optional[Path] = None
    # Watch/auto-reindex is configured PER SOURCE (a `watch:` block on each source),
    # not instance-wide — see basic_kb.watcher.resolve_settings.


def load_config(config_path: Path) -> Config:
    """Parse an instance config YAML. Raises on missing file or empty sources."""
    import yaml

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    stem = config_path.stem  # e.g. "basic-kb"
    for suffix in _LOCAL_SUFFIXES:
        local_path = config_path.with_name(stem + suffix)
        if local_path.exists():
            local_data = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
            _deep_merge(data, local_data)
            break

    base_dir = config_path.parent
    chunker_cfg = data.get("chunker", {}) or {}

    sources = data.get("sources", []) or []
    if not sources:
        raise ValueError(f"Config {config_path} defines no `sources:`.")

    env_file = _resolve(base_dir, data["env_file"]) if data.get("env_file") else None

    # `reranker:` may be a string ("local") or a mapping ({type, model, candidates}).
    rr = data.get("reranker")
    cand: dict = {}
    if isinstance(rr, str):
        reranker_type, reranker_model = rr.strip(), None
    elif isinstance(rr, dict):
        reranker_type, reranker_model = str(rr.get("type", "none")), rr.get("model")
        cand = rr.get("candidates", {}) or {}
    else:
        reranker_type, reranker_model = "none", None

    fresh = data.get("freshness", {}) or {}

    thr = data.get("throttle", {}) or {}
    thr_cores = thr.get("cores_fraction")

    # `reindex_guard:` may be a bare bool (reindex_guard: false) or a mapping
    # ({enabled, threshold}). Both disable/tune the mass-change corruption check.
    rg = data.get("reindex_guard")
    if isinstance(rg, bool):
        rg_enabled, rg_threshold = rg, 0.9
    else:
        rg = rg or {}
        rg_enabled = bool(rg.get("enabled", True))
        rg_threshold = float(rg.get("threshold", 0.9))

    return Config(
        path=config_path,
        base_dir=base_dir,
        name=data.get("name", config_path.stem),
        store_dir=_resolve(base_dir, data.get("store_dir", ".chroma")),
        embedding_model=data.get("embedding_model", DEFAULT_MODEL),
        chunk_size=int(chunker_cfg.get("max_chunk_size", DEFAULT_CHUNK_SIZE)),
        overlap=int(chunker_cfg.get("overlap", DEFAULT_OVERLAP)),
        min_chunk=int(chunker_cfg.get("min_chunk_size", DEFAULT_MIN_CHUNK)),
        sources=sources,
        reranker_type=reranker_type,
        reranker_model=reranker_model,
        cand_multiplier=int(cand.get("multiplier", 3)),
        cand_min=int(cand.get("min", 50)),
        cand_max=int(cand.get("max", 200)),
        timing=bool(data.get("timing", False)),
        freshness_enabled=bool(fresh.get("enabled", True)),
        freshness_every_days=int(fresh.get("every_days", 3)),
        freshness_message=str(fresh.get("message", DEFAULT_FRESHNESS_MESSAGE)),
        throttle_cores=float(thr_cores) if thr_cores is not None else None,
        throttle_priority=str(thr.get("priority", "normal")),
        throttle_pause_ms=int(thr.get("pause_ms", 0)),
        throttle_pause_every=int(thr.get("pause_every", 50)),
        log_file=_resolve(base_dir, data["log_file"]) if data.get("log_file") else None,
        log_level=str(data.get("log_level", "INFO")),
        log_max_bytes=int(data.get("log_max_bytes", 10_000_000)),
        log_backup_count=int(data.get("log_backup_count", 5)),
        reindex_guard=rg_enabled,
        reindex_guard_threshold=rg_threshold,
        env_file=env_file,
    )
