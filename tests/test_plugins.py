"""External source types loaded by import path (ADR 0005), and search-time dedup by `dedup_key`."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from basic_kb.core import KnowledgeBase
from basic_kb.plugins import load_import_path
from basic_kb.sources import build_source
from basic_kb.transforms import RedactSecrets

from .fakes import FakeEmbedder

# A source type living next to the instance config, the way an external consumer ships one.
# Each line of a .txt file is a chunk; the line itself is the dedup key, so the same line in
# two files is "the same content" — the shape of a conversation turn replayed by a fork.
PLUGIN = textwrap.dedent("""\
    from basic_kb.chunkers import ChunkerBase
    from basic_kb.models import Chunk, ParsedDocument
    from basic_kb.sources import DataSourceBase


    class LineChunker(ChunkerBase):
        name = "LineChunker"

        def chunk(self, doc):
            lines = [l for l in doc.body.splitlines() if l.strip()]
            return [Chunk(text=l, metadata={**doc.base_metadata, "dedup_key": l}, id_suffix=str(i))
                    for i, l in enumerate(lines)]


    class LineSource(DataSourceBase):
        type_name = "lines"

        def __init__(self, source_id, directory, label="", description="", chunker="lines",
                     exclude=None, greeting="hello"):
            super().__init__(source_id, directory, label, description, chunker, exclude)
            self.greeting = greeting

        def get_files(self):
            return self._keep_files(sorted(self._dir.glob("*.txt")))

        def parse_file(self, path):
            return ParsedDocument(id=path.stem, title=path.stem, date="unknown",
                                  body=path.read_text(encoding="utf-8"),
                                  base_metadata={"source": self._id, "file": path.name})

        def make_chunker(self, chunk_size, overlap, min_chunk):
            return LineChunker()


    class NotASource:
        pass
    """)


@pytest.fixture
def plugin_dir(tmp_path: Path) -> tuple[Path, str]:
    """An instance folder holding a uniquely named plugin module (sys.modules caches by name)."""
    module = f"lines_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module}.py").write_text(PLUGIN, encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "original.txt").write_text("the grinder broke on monday\nwe ordered a new kettle\n", encoding="utf-8")
    (data / "fork.txt").write_text("the grinder broke on monday\nthe fork added a reply about tea\n", encoding="utf-8")
    return tmp_path, module


def test_import_path_builds_external_source_with_its_own_options(plugin_dir):
    base, module = plugin_dir
    src = build_source({"id": "l", "type": f"{module}:LineSource", "path": "data", "greeting": "hej"}, base)
    assert type(src).__name__ == "LineSource" and src.greeting == "hej"
    assert [f.name for f in src.get_files()] == ["fork.txt", "original.txt"]


def test_unknown_option_names_the_source(plugin_dir):
    base, module = plugin_dir
    with pytest.raises(ValueError, match="'l'.*colour"):
        build_source({"id": "l", "type": f"{module}:LineSource", "path": "data", "colour": "red"}, base)


@pytest.mark.parametrize("type_value, message", [
    ("no_such_module_xyz:Thing", "cannot import module 'no_such_module_xyz'"),
    ("{module}:Missing", "has no attribute 'Missing'"),
    ("{module}:NotASource", "not a subclass"),
    ("{module}:", "package.module:Name"),
])
def test_bad_import_paths_fail_loudly_naming_the_path(plugin_dir, type_value, message):
    base, module = plugin_dir
    type_value = type_value.format(module=module)
    with pytest.raises(ValueError, match=message) as err:
        build_source({"id": "l", "type": type_value, "path": "data"}, base)
    assert type_value in str(err.value)


def test_short_names_still_resolve_through_the_registry(tmp_path):
    assert type(build_source({"id": "m", "type": "markdown", "path": "d"}, tmp_path)).__name__ == "MarkdownSource"


def test_load_import_path_resolves_dotted_attributes():
    assert load_import_path("basic_kb.transforms:RedactSecrets.name") == "redact-secrets"


def test_transforms_attach_to_external_sources(plugin_dir):
    base, module = plugin_dir
    src = build_source({"id": "l", "type": f"{module}:LineSource", "path": "data",
                        "transforms": ["redact-secrets"]}, base)
    assert [type(t) for t in src.transforms] == [RedactSecrets]


def test_same_content_in_two_files_surfaces_once(plugin_dir):
    base, module = plugin_dir
    src = build_source({"id": "l", "type": f"{module}:LineSource", "path": "data"}, base)
    kb = KnowledgeBase(embedder=FakeEmbedder(), store_dir=base / ".basic-kb")
    result = kb.index(src)
    assert result.added == 2 and kb.store.count("l") == 4      # both copies are stored...
    hits = kb.search([src], ["grinder broke monday"], n=10)
    docs = [h.doc for h in hits]
    assert docs.count("the grinder broke on monday") == 1       # ...and found once
    assert len(docs) == 3
    assert all("dedup_key" not in h.trimmed().metadata for h in hits)


def test_dedup_widens_k_to_fill_n(plugin_dir):
    base, module = plugin_dir
    data = base / "data"
    for i in range(6):
        (data / f"copy{i}.txt").write_text("the grinder broke on monday\n", encoding="utf-8")
    src = build_source({"id": "l", "type": f"{module}:LineSource", "path": "data"}, base)
    kb = KnowledgeBase(embedder=FakeEmbedder(), store_dir=base / ".basic-kb")
    kb.index(src)
    # Eight copies of one line rank above everything else; n=3 still finds three distinct hits.
    assert len(kb.search([src], ["grinder broke monday"], n=3)) == 3
