"""Instance config: loading, local overrides, env files, discovery."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from basic_kb.config import DEFAULT_FRESHNESS_MESSAGE, find_config, load_config, load_env_file


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


MINIMAL = "sources:\n  - id: a\n    path: data\n"


def yaml_for(extra: str = "") -> str:
    """MINIMAL plus an extra block; both dedented so the YAML stays valid."""
    return MINIMAL + textwrap.dedent(extra)


def test_fixture_config_values(config, instance):
    assert config.name == "test-instance"
    assert config.base_dir == instance
    assert config.store_dir == instance / ".basic-kb"
    assert (config.chunk_size, config.overlap, config.min_chunk) == (400, 40, 20)
    assert [s["id"] for s in config.sources] == ["notes", "meetings"]
    assert config.freshness_enabled is False
    assert config.vacuum_enabled is False


def test_defaults_when_keys_absent(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", MINIMAL))
    assert cfg.name == "basic-kb"                      # config file stem
    assert cfg.store_dir == tmp_path / ".basic-kb"
    assert cfg.embedding_model == "bge-small-en-v1.5"
    assert (cfg.chunk_size, cfg.overlap, cfg.min_chunk) == (1500, 150, 50)
    assert cfg.reranker_type == "none" and cfg.reranker_model is None
    assert (cfg.cand_multiplier, cfg.cand_min, cfg.cand_max) == (3, 50, 200)
    assert cfg.search_n is None and cfg.search_separate is False
    assert cfg.freshness_enabled is True
    assert cfg.freshness_message == DEFAULT_FRESHNESS_MESSAGE
    assert cfg.throttle_priority == "normal" and cfg.throttle_cores is None
    assert cfg.reindex_guard is True and cfg.reindex_guard_threshold == 0.9
    assert cfg.env_file_search_up == 5
    assert cfg.log_file is None
    assert cfg.embed_batch_size == 8
    assert (cfg.serve_host, cfg.serve_port, cfg.serve_auth, cfg.serve_watch) == ("127.0.0.1", 8765, False, False)


def test_serve_block(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("""\
        serve:
          host: 0.0.0.0
          port: 9000
          auth: true
          watch: true
        """)))
    assert (cfg.serve_host, cfg.serve_port, cfg.serve_auth, cfg.serve_watch) == ("0.0.0.0", 9000, True, True)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_no_sources_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="sources"):
        load_config(write(tmp_path / "basic-kb.yaml", "name: x\n"))


def test_local_override_deep_merges_and_replaces_lists(tmp_path: Path):
    write(tmp_path / "basic-kb.yaml", """\
        chunker:
          max_chunk_size: 400
          overlap: 40
        sources:
          - id: a
            path: data/a
          - id: b
            path: data/b
        """)
    write(tmp_path / "basic-kb.local.yaml", """\
        chunker:
          overlap: 7
        sources:
          - id: only
            path: data/only
        """)
    cfg = load_config(tmp_path / "basic-kb.yaml")
    assert cfg.chunk_size == 400          # kept from base
    assert cfg.overlap == 7               # overridden
    assert [s["id"] for s in cfg.sources] == ["only"]


def test_reranker_as_string(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("reranker: local\n")))
    assert cfg.reranker_type == "local" and cfg.reranker_model is None


def test_reranker_mapping_splits_options_and_candidates(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("""\
        reranker:
          type: jina-compatible
          model: rerank-3
          base_url: https://api.voyageai.com/v1/rerank
          api_key_env: VOYAGE_API_KEY
          top_k_param: top_k
          candidates:
            multiplier: 2
            min: 10
            max: 40
        """)))
    assert cfg.reranker_type == "jina-compatible"
    assert cfg.reranker_model == "rerank-3"
    assert cfg.reranker_options == {"base_url": "https://api.voyageai.com/v1/rerank",
                                    "api_key_env": "VOYAGE_API_KEY", "top_k_param": "top_k"}
    assert (cfg.cand_multiplier, cfg.cand_min, cfg.cand_max) == (2, 10, 40)


def test_bare_bool_blocks(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("vacuum: false\nreindex_guard: false\n")))
    assert cfg.vacuum_enabled is False
    assert cfg.vacuum_deleted_fraction == 0.2 and cfg.vacuum_min_deleted == 1000
    assert cfg.reindex_guard is False and cfg.reindex_guard_threshold == 0.9


def test_mapping_blocks(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("""\
        vacuum:
          enabled: true
          deleted_fraction: 0.5
          min_deleted: 10
        reindex_guard:
          threshold: 0.75
        throttle:
          cores_fraction: 0.25
          priority: low
          pause_ms: 100
          pause_every: 5
        """)))
    assert (cfg.vacuum_enabled, cfg.vacuum_deleted_fraction, cfg.vacuum_min_deleted) == (True, 0.5, 10)
    assert (cfg.reindex_guard, cfg.reindex_guard_threshold) == (True, 0.75)
    assert (cfg.throttle_cores, cfg.throttle_priority, cfg.throttle_pause_ms, cfg.throttle_pause_every) == (0.25, "low", 100, 5)


def test_search_block_and_legacy_keys(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("""\
        timing: true
        search:
          n: 7
          batch: true
          max_chars: 300
          content_type: help
        freshness:
          every_days: 9
        """)))
    assert cfg.search_n == 7
    assert cfg.search_separate is True          # `batch` alias
    assert cfg.search_max_chars == 300
    assert cfg.search_content_type == "help"
    assert cfg.search_timing is True            # legacy top-level `timing`
    assert cfg.freshness_stale_after_days == 9  # legacy `every_days`


def test_log_file_and_embedding_block(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("""\
        log_file: logs/kb.log
        log_backup_count: 3
        embedding:
          provider: openai-compatible
          base_url: https://x/v1
        """)))
    assert cfg.log_file == tmp_path / "logs" / "kb.log"
    assert cfg.log_backup_count == 3
    assert cfg.embedding == {"provider": "openai-compatible", "base_url": "https://x/v1"}


# --- env file resolution ------------------------------------------------------------------

def test_explicit_env_file_that_exists(tmp_path: Path):
    write(tmp_path / "secrets.env", "A=1\n")
    cfg = load_config(write(tmp_path / "inst" / "basic-kb.yaml", yaml_for("env_file: ../secrets.env\n")))
    assert cfg.env_file.resolve() == (tmp_path / "secrets.env").resolve()


def test_env_file_walks_up_to_nearest_dotenv(tmp_path: Path):
    write(tmp_path / ".env", "A=root\n")
    cfg = load_config(write(tmp_path / "a" / "b" / "c" / "basic-kb.yaml", MINIMAL))
    assert cfg.env_file == tmp_path / ".env"


def test_env_file_walk_up_prefers_nearest(tmp_path: Path):
    write(tmp_path / ".env", "A=root\n")
    write(tmp_path / "a" / ".env", "A=near\n")
    cfg = load_config(write(tmp_path / "a" / "b" / "basic-kb.yaml", MINIMAL))
    assert cfg.env_file == tmp_path / "a" / ".env"


def test_env_file_search_up_zero_keeps_declared_path(tmp_path: Path):
    write(tmp_path / ".env", "A=1\n")
    cfg = load_config(write(tmp_path / "a" / "basic-kb.yaml", yaml_for("env_file: missing.env\nenv_file_search_up: 0\n")))
    assert cfg.env_file == tmp_path / "a" / "missing.env"
    assert not cfg.env_file.exists()


def test_load_env_file_setdefault_and_quotes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KEEP", "shell")
    monkeypatch.delenv("NEWKEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    p = write(tmp_path / ".env", """\
        # comment
        KEEP=file
        NEWKEY=value
        QUOTED="with spaces"
        garbage line
        """)
    assert load_env_file(p) == 3
    assert os.environ["KEEP"] == "shell"
    assert os.environ["NEWKEY"] == "value"
    assert os.environ["QUOTED"] == "with spaces"


# --- discovery -------------------------------------------------------------------------------

def test_find_config_env_var_wins(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BASIC_KB_CONFIG", str(tmp_path / "x.yaml"))
    assert find_config(tmp_path) == tmp_path / "x.yaml"


def test_find_config_walks_up(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    cfg = write(tmp_path / "basic-kb.yaml", MINIMAL)
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_config(deep) == cfg


def test_find_config_none_when_absent(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BASIC_KB_CONFIG", raising=False)
    assert find_config(tmp_path) is None


def test_attach_cli_block(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml", yaml_for("""\
        attach_cli:
          url: https://pkb.example.com
          key_env: PKB_KEY
          key_file: keys/local.key
        """)))
    assert cfg.attach_url == "https://pkb.example.com"
    assert cfg.attach_key_env == "PKB_KEY"
    assert cfg.attach_key_file == tmp_path / "keys" / "local.key"

    bare = load_config(write(tmp_path / "b" / "basic-kb.yaml", yaml_for("attach_cli: https://x\n")))
    assert bare.attach_url == "https://x" and bare.attach_key_env is None


def test_a_client_config_may_omit_sources(tmp_path: Path):
    cfg = load_config(write(tmp_path / "basic-kb.yaml",
                            "name: client\nattach_cli:\n  url: https://pkb.example.com\n"))
    assert cfg.sources == [] and cfg.attach_url == "https://pkb.example.com"


def test_the_old_attach_key_is_refused_rather_than_ignored(tmp_path: Path):
    """Silently dropping it would send every command to the local store instead."""
    with pytest.raises(ValueError, match="now `attach_cli:`"):
        load_config(write(tmp_path / "basic-kb.yaml", yaml_for("attach:\n  url: https://x\n")))


def test_a_literal_key_in_the_config_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="never live in a config file"):
        load_config(write(tmp_path / "basic-kb.yaml", yaml_for("attach_cli:\n  url: https://x\n  key: secret\n")))
