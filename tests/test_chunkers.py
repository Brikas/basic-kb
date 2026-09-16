"""Chunkers: recursive and breadcrumb, plus the registry."""
from __future__ import annotations

import pytest

from basic_kb.chunkers import BreadcrumbHeadingChunker, RecursiveChunker, build_chunker
from basic_kb.models import ParsedDocument


def doc(body: str, title: str = "Page") -> ParsedDocument:
    return ParsedDocument(id="d1", title=title, date="unknown", body=body,
                          base_metadata={"title": title, "source": "s"})


# --- registry ------------------------------------------------------------------------

def test_build_chunker_by_name():
    assert isinstance(build_chunker("recursive", 100, 10, 5), RecursiveChunker)
    assert isinstance(build_chunker("breadcrumb", 100, 10, 5), BreadcrumbHeadingChunker)


def test_build_chunker_unknown_name_raises_with_options():
    with pytest.raises(ValueError, match="breadcrumb"):
        build_chunker("nope", 100, 10, 5)


def test_chunker_names():
    assert build_chunker("recursive", 100, 10, 5).name == "RecursiveChunker"
    assert build_chunker("breadcrumb", 100, 10, 5).name == "BreadcrumbHeadingChunker"


# --- recursive -----------------------------------------------------------------------

def test_recursive_chunks_carry_base_metadata_and_index():
    body = "\n\n".join(f"Paragraph {i} " + "x" * 60 for i in range(5))
    chunks = RecursiveChunker(100, 0, 5).chunk(doc(body))
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.metadata["title"] == "Page"
        assert c.metadata["source"] == "s"
        assert c.metadata["chunk_index"] == i
        assert c.id_suffix == str(i)
        assert len(c.text) <= 100


# --- breadcrumb ----------------------------------------------------------------------

BODY = """# Coffee

Intro paragraph about coffee that is long enough to keep.

![hero](hero.png)

## Ratios

Sixteen to one is the usual starting ratio for a pour over.

## Gear

A burr grinder matters more than the kettle.
"""


def test_breadcrumb_intro_and_sections():
    chunks = BreadcrumbHeadingChunker(400, 10, 20).chunk(doc(BODY, title="fallback"))
    texts = [c.text for c in chunks]
    assert texts[0].startswith("# Coffee\n\nIntro paragraph")
    assert texts[1].startswith("# Coffee > ## Ratios\n\nSixteen")
    assert texts[2].startswith("# Coffee > ## Gear\n\nA burr")
    assert [c.id_suffix for c in chunks] == ["0", "1", "2"]


def test_breadcrumb_metadata_has_headings():
    chunks = BreadcrumbHeadingChunker(400, 10, 20).chunk(doc(BODY))
    assert chunks[0].metadata["breadcrumb"] == "# Coffee"
    assert "h2" not in chunks[0].metadata
    assert chunks[1].metadata["h1"] == "Coffee"
    assert chunks[1].metadata["h2"] == "Ratios"
    assert chunks[1].metadata["breadcrumb"] == "# Coffee > ## Ratios"


def test_breadcrumb_strips_image_lines():
    chunks = BreadcrumbHeadingChunker(400, 10, 20).chunk(doc(BODY))
    assert "hero.png" not in "".join(c.text for c in chunks)


def test_breadcrumb_without_h2_is_one_chunk_under_h1():
    chunks = BreadcrumbHeadingChunker(400, 10, 5).chunk(doc("# Only\n\nJust one body here."))
    assert len(chunks) == 1
    assert chunks[0].text == "# Only\n\nJust one body here."


def test_breadcrumb_falls_back_to_doc_title_without_h1():
    chunks = BreadcrumbHeadingChunker(400, 10, 5).chunk(doc("Body with no heading at all.", title="T"))
    assert chunks[0].text.startswith("# T\n\n")


def test_breadcrumb_heading_becomes_body_when_section_is_empty():
    body = "# Dict\n\n## A definition long enough to stand alone as text\n\n## no\n\n"
    chunks = BreadcrumbHeadingChunker(400, 10, 20).chunk(doc(body))
    assert len(chunks) == 1
    assert chunks[0].text.endswith("\n\nA definition long enough to stand alone as text")


def test_breadcrumb_oversized_section_splits_and_keeps_prefix():
    # The split budget is floored at 200 chars, so a max below ~220 can be exceeded;
    # realistic sizes (>= 400) always hold the bound.
    body = "# T\n\n## Big\n\n" + " ".join(f"word{i}" for i in range(400))
    chunks = BreadcrumbHeadingChunker(600, 10, 20).chunk(doc(body))
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("# T > ## Big\n\n")
        assert len(c.text) <= 600
    assert [c.id_suffix for c in chunks] == [str(i) for i in range(len(chunks))]


def test_breadcrumb_oversized_section_is_flagged_in_metadata():
    body = "# T\n\n## Big\n\n" + " ".join(f"word{i}" for i in range(400))
    chunks = BreadcrumbHeadingChunker(600, 10, 20).chunk(doc(body))
    assert len(chunks) > 1 and all(c.metadata["oversized"] is True for c in chunks)
    small = BreadcrumbHeadingChunker(600, 10, 20).chunk(doc("# T\n\n## S\n\nA short section that fits."))
    assert "oversized" not in small[0].metadata


def test_breadcrumb_drops_sections_below_min():
    body = "# T\n\n## tiny\n\nno\n\n## Real section\n\nThis one has enough body text to keep."
    chunks = BreadcrumbHeadingChunker(400, 10, 20).chunk(doc(body))
    assert [c.metadata["h2"] for c in chunks] == ["Real section"]
