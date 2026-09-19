"""Shared fixtures: a throwaway instance on disk and a KnowledgeBase over a fake embedder.

Everything here is offline. The store is the real sqlite-vec store in a temp dir;
only the embedding model is replaced (see fakes.py).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from basic_kb.config import Config, load_config
from basic_kb.core import KnowledgeBase
from basic_kb.embedders import EMBEDDER_PROVIDERS
from basic_kb.sources import build_source

from .fakes import FakeEmbedder


# --- sample documents -------------------------------------------------------------

NOTES = {
    "coffee.md": textwrap.dedent("""\
        ---
        title: Coffee brewing
        content_type: hobby
        url: https://example.test/coffee
        ---
        # Coffee brewing

        Grind size matters more than water temperature for a pour over.

        ## Ratios

        Use sixteen grams of water per gram of coffee for a balanced cup.

        ## Equipment

        A gooseneck kettle and a burr grinder are the two purchases that matter.
        """),
    "taxes.md": textwrap.dedent("""\
        ---
        title: Tax deadlines
        content_type: admin
        ---
        # Tax deadlines

        The annual return is due in March; quarterly VAT is due the month after each quarter.

        ## Receipts

        Keep receipts for seven years in case of an audit.
        """),
    "sub/travel.md": textwrap.dedent("""\
        ---
        title: Travel packing
        content_type: hobby
        ---
        # Travel packing

        Pack the charger first; everything else can be bought at the destination.
        """),
    "drafts/secret.md": "# Draft\n\nThis file is excluded by pattern and must never be indexed.\n",
    "stub.md": "# Stub\n\ntiny\n",   # too short to chunk: tracked, no chunks
}

TRANSCRIPTS = {
    "2026-01-05-standup.md": textwrap.dedent("""\
        # Weekly standup
        *2026-01-05*

        **Alice:** We shipped the pricing page and the budget concern from the client is resolved.

        **Bob:** The price was too high for the pilot; we agreed a discount for the first quarter.

        Some narration line without a speaker tag that must be dropped.
        """),
    "2026-02-10-retro.md": textwrap.dedent("""\
        # Sprint retro
        *2026-02-10*

        **Alice:** Coffee in the office got better after we bought a burr grinder.
        """),
}


def write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


CONFIG_YAML = textwrap.dedent("""\
    name: test-instance
    store_dir: .basic-kb
    embedding_model: fake-bow-64
    chunker:
      max_chunk_size: 400
      overlap: 40
      min_chunk_size: 20
    freshness:
      enabled: false
    vacuum:
      enabled: false
    sources:
      - id: notes
        type: markdown
        path: data/notes
        chunker: breadcrumb
        label: Test Notes
        description: Markdown notes with frontmatter.
        exclude:
          - "drafts/"
      - id: meetings
        type: transcript
        path: data/meetings
        chunker: recursive
        label: Meetings
        description: Speaker-tagged transcripts.
    """)


@pytest.fixture
def instance(tmp_path: Path) -> Path:
    """A complete instance folder: basic-kb.yaml plus two populated sources."""
    write_tree(tmp_path / "data" / "notes", NOTES)
    write_tree(tmp_path / "data" / "meetings", TRANSCRIPTS)
    (tmp_path / "basic-kb.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def config(instance: Path) -> Config:
    return load_config(instance / "basic-kb.yaml")


@pytest.fixture
def sources(config: Config):
    return [build_source(s, config.base_dir) for s in config.sources]


@pytest.fixture
def notes(sources):
    return sources[0]


@pytest.fixture
def meetings(sources):
    return sources[1]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def kb(config: Config, embedder: FakeEmbedder) -> KnowledgeBase:
    return KnowledgeBase(embedder=embedder, store_dir=config.store_dir)


@pytest.fixture
def indexed_kb(kb: KnowledgeBase, sources) -> KnowledgeBase:
    """A KnowledgeBase with both sources fully indexed."""
    for s in sources:
        r = kb.index(s, chunk_size=400, overlap=40, min_chunk=20)
        assert not r.aborted, r.abort_reason
    return kb


@pytest.fixture
def fake_provider(monkeypatch):
    """Make `KnowledgeBase.from_config` build the fake embedder: the fixture config names
    no provider, so `local` is what it resolves to."""
    monkeypatch.setitem(EMBEDDER_PROVIDERS, "local", lambda cfg, threads=None: FakeEmbedder())


@pytest.fixture
def served(indexed_kb, config, fake_provider):
    """A live `KBServer` for the indexed fixture instance, on a free loopback port, auth off.
    Yields the server; `served.url` is where it listens."""
    from basic_kb.server import KBServer

    server = KBServer(config, port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def served_auth(indexed_kb, config, fake_provider):
    """Same, with authentication on and no API keys minted yet."""
    from basic_kb.server import KBServer

    server = KBServer(config, port=0, auth=True)
    server.start()
    try:
        yield server
    finally:
        server.stop()
