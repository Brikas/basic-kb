"""Sources: file discovery, parsing, exclusion patterns, registry."""
from __future__ import annotations

from pathlib import Path

import pytest

from basic_kb.errors import UnknownSource
from basic_kb.sources import (
    MarkdownSource, TranscriptSource, _parse_frontmatter, build_source, parse_transcript, path_excluded,
    resolve_sources,
)

from .conftest import TRANSCRIPTS, write_tree


# --- frontmatter -----------------------------------------------------------------------

def test_frontmatter_parsed_and_body_stripped():
    meta, body = _parse_frontmatter("---\ntitle: X\ncontent_type: a\n---\n\nBody here\n")
    assert meta == {"title": "X", "content_type": "a"}
    assert body == "Body here"


def test_no_frontmatter_passes_through():
    assert _parse_frontmatter("Just text") == ({}, "Just text")


def test_unterminated_frontmatter_is_treated_as_body():
    text = "---\ntitle: X\nno end"
    assert _parse_frontmatter(text) == ({}, text)


def test_invalid_yaml_frontmatter_gives_empty_meta():
    meta, body = _parse_frontmatter("---\n: : [\n---\nBody")
    assert meta == {}
    assert body == "Body"


# --- transcripts -----------------------------------------------------------------------

def test_parse_transcript_title_date_and_speaker_lines(tmp_path: Path):
    p = tmp_path / "t.md"
    p.write_text(TRANSCRIPTS["2026-01-05-standup.md"], encoding="utf-8")
    title, date, body = parse_transcript(p)
    assert title == "Weekly standup"
    assert date == "2026-01-05"
    paragraphs = body.split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("**Alice:**")
    assert "narration" not in body


def test_parse_transcript_without_date(tmp_path: Path):
    p = tmp_path / "t.md"
    p.write_text("# Untitled\n\n**A:** hi\n", encoding="utf-8")
    assert parse_transcript(p)[1] == "unknown"


# --- exclusion patterns ----------------------------------------------------------------

@pytest.mark.parametrize("rel, patterns, expected", [
    ("a/b/x.tmp", ["*.tmp"], True),
    ("x.tmp", ["*.tmp"], True),
    ("a/b/x.md", ["*.tmp"], False),
    ("notes/TODO-list.md", ["TODO*"], True),
    ("drafts/x.md", ["drafts/"], True),
    ("a/drafts/x.md", ["drafts/"], True),
    ("drafts", ["drafts/"], False),               # a file named drafts is not the folder
    ("archive/old-1.md", ["archive/old*"], True),
    ("x/archive/old-1.md", ["archive/old*"], False),  # anchored at the root
    ("keep.md", ["*.md", "!keep.md"], False),
    ("other.md", ["*.md", "!keep.md"], True),
    ("x.md", ["", "# comment", "*.md"], True),
    ("x.md", [], False),
])
def test_path_excluded(rel, patterns, expected):
    assert path_excluded(rel, patterns) is expected


def test_leading_slash_anchors_at_root():
    assert path_excluded("notes.md", ["/notes.md"]) is True
    assert path_excluded("sub/notes.md", ["/notes.md"]) is False


# --- registry / factory -----------------------------------------------------------------

def test_build_source_unknown_type_lists_options(tmp_path: Path):
    with pytest.raises(ValueError, match="markdown"):
        build_source({"id": "x", "type": "pdf", "path": "d"}, tmp_path)


def test_build_source_resolves_relative_path_against_base_dir(tmp_path: Path):
    src = build_source({"id": "n", "path": "data/n"}, tmp_path)
    assert src.directory == tmp_path / "data" / "n"
    absolute = build_source({"id": "n", "path": str(tmp_path / "abs")}, tmp_path)
    assert absolute.directory == tmp_path / "abs"


def test_build_source_defaults(tmp_path: Path):
    md = build_source({"id": "n", "path": "d"}, tmp_path)
    assert isinstance(md, MarkdownSource)
    assert md.chunker_name == "breadcrumb"
    assert md.label == "n"
    tr = build_source({"id": "t", "type": "transcript", "path": "d"}, tmp_path)
    assert isinstance(tr, TranscriptSource)
    assert tr.chunker_name == "recursive"


def test_build_source_content_type_filter_only_for_markdown(tmp_path: Path):
    md = build_source({"id": "n", "path": "d"}, tmp_path, content_type_filter="hobby")
    assert md.label == "n [hobby]"
    tr = build_source({"id": "t", "type": "transcript", "path": "d"}, tmp_path, content_type_filter="hobby")
    assert tr.label == "t"


def test_resolve_sources_selectors(config):
    assert [s.source_id for s in resolve_sources(config)] == ["notes", "meetings"]
    assert [s.source_id for s in resolve_sources(config, "all")] == ["notes", "meetings"]
    assert [s.source_id for s in resolve_sources(config, "meetings, notes")] == ["meetings", "notes"]
    assert [s.source_id for s in resolve_sources(config, ["notes"])] == ["notes"]
    assert resolve_sources(config, "notes", content_type_filter="hobby")[0].label == "Test Notes [hobby]"


def test_resolve_sources_reports_every_unknown_id(config):
    with pytest.raises(UnknownSource) as e:
        resolve_sources(config, "notes,nope,zip")
    assert e.value.requested == ["nope", "zip"] and e.value.known == ["notes", "meetings"]
    assert "Unknown source 'nope', 'zip'. Configured: notes, meetings" in str(e.value)


# --- MarkdownSource ---------------------------------------------------------------------

def test_markdown_get_files_recursive_sorted_and_excluded(notes):
    names = [f.relative_to(notes.directory).as_posix() for f in notes.get_files()]
    assert names == ["coffee.md", "stub.md", "sub/travel.md", "taxes.md"]


def test_markdown_content_type_filter_reads_frontmatter(notes, config):
    hobby = build_source(config.sources[0], config.base_dir, content_type_filter="hobby")
    names = sorted(f.name for f in hobby.get_files())
    assert names == ["coffee.md", "travel.md"]


def test_markdown_parse_file_lifts_frontmatter(notes):
    d = notes.parse_file(notes.directory / "coffee.md")
    assert d.title == "Coffee brewing"
    assert d.id == "coffee"
    assert d.date == "unknown"
    assert d.base_metadata == {
        "title": "Coffee brewing", "url": "https://example.test/coffee",
        "content_type": "hobby", "file": "coffee.md", "source": "notes",
    }
    assert d.body.startswith("# Coffee brewing")


def test_markdown_parse_file_empty_body_is_none(tmp_path: Path):
    src = MarkdownSource("n", tmp_path)
    p = tmp_path / "e.md"
    p.write_text("---\ntitle: Empty\n---\n\n", encoding="utf-8")
    assert src.parse_file(p) is None


def test_markdown_missing_directory_has_no_files(tmp_path: Path):
    assert MarkdownSource("n", tmp_path / "missing").get_files() == []


def test_is_excluded_outside_directory_matches_bare_name(tmp_path: Path):
    src = MarkdownSource("n", tmp_path / "d", exclude=["*.private.md"])
    assert src.is_excluded(Path("/elsewhere/x.private.md")) is True
    assert src.is_excluded(Path("/elsewhere/x.md")) is False


# --- TranscriptSource -------------------------------------------------------------------

def test_transcript_get_files_is_flat(meetings, tmp_path: Path):
    write_tree(meetings.directory / "nested", {"deep.md": "# x\n\n**A:** hi\n"})
    names = [f.name for f in meetings.get_files()]
    assert names == ["2026-01-05-standup.md", "2026-02-10-retro.md"]


def test_transcript_parse_file(meetings):
    d = meetings.parse_file(meetings.directory / "2026-01-05-standup.md")
    assert d.title == "Weekly standup"
    assert d.date == "2026-01-05"
    assert d.base_metadata["source"] == "meetings"
    assert d.base_metadata["file"] == "2026-01-05-standup.md"


def test_transcript_without_speaker_lines_is_none(meetings, tmp_path: Path):
    p = meetings.directory / "plain.md"
    p.write_text("# Plain\n\nNo speakers here.\n", encoding="utf-8")
    assert meetings.parse_file(p) is None
