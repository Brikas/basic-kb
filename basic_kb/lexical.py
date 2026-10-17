"""Keyword side of hybrid search: turn a natural-language query into a safe FTS5 MATCH.

The store indexes chunk text with FTS5's `unicode61` tokenizer, which keeps letters and
digits and splits on everything else — `_`, `-`, `.`, `/`, `:`. A query has to produce the
same tokens the index holds, so a term like `basic_kb/core.py` or `--switch-model` becomes
the phrase of its parts (`"basic kb core py"`), which matches exactly where those tokens
sit side by side. A plain word becomes a prefix term (`"deploy"*` finds `deployed`).

Terms are joined with OR, so BM25 ranks chunks by how many rare terms they hold instead of
dropping every chunk that misses one word: queries here are agent-written sentences, and an
AND of a sentence matches almost nothing. Common English words are dropped unless nothing
else is left. A `"quoted phrase"` stays one phrase.

Every part is quoted, so no input can be read as FTS5 syntax (`OR`, `NOT`, `col:`, `*`,
`^`, unbalanced quotes). Oracles: tobi/qmd `buildFTS5Query`, simonw/sqlite-utils
`quote_fts`, ariel-frischer/kb `fts_escape` (see SPEC.md).
"""
from __future__ import annotations

import re
from typing import Optional

# Must match the store's FTS5 table definition (store.py) — the query splits terms the
# way this tokenizer splits text.
FTS_TOKENIZER = "unicode61 remove_diacritics 2"

# What unicode61 keeps as token characters: letters and digits. `_` is punctuation to it.
_PART = re.compile(r"[^\W_]+")
_TERM = re.compile(r'"([^"]*)"|(\S+)')

# Prefix matching on very short parts expands to most of the vocabulary and slows the
# query for no gain; below this length a part is matched whole.
PREFIX_MIN_CHARS = 3
# A runaway query (a pasted paragraph) would OR hundreds of terms; past this many the rest
# add cost and almost no ranking signal.
MAX_TERMS = 32

STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does doing for from had has have "
    "how i if in into is it its me my of on or our so than that the their them then there "
    "these they this to was we were what when where which while who why will with would "
    "you your".split()
)


def build_match(query: str) -> Optional[str]:
    """FTS5 MATCH expression for a free-text query, or None when it holds no searchable term.

    >>> build_match("how did I fix basic_kb/core.py")
    '"fix"* OR "basic kb core py"'
    """
    if query.count('"') % 2:
        query += '"'                       # balance a stray quote instead of failing
    terms: list[str] = []
    stopped: list[str] = []
    for m in _TERM.finditer(query):
        quoted, bare = m.group(1), m.group(2)
        parts = _PART.findall((quoted if quoted is not None else bare).lower())
        if not parts:
            continue
        noise = all(p in STOPWORDS or len(p) < 2 for p in parts)    # "the", "a", "what's"
        if quoted is not None or len(parts) > 1:
            term = '"' + " ".join(parts) + '"'
        else:
            term = f'"{parts[0]}"*' if len(parts[0]) >= PREFIX_MIN_CHARS else f'"{parts[0]}"'
        (stopped if noise and quoted is None else terms).append(term)
    chosen = list(dict.fromkeys(terms or stopped))[:MAX_TERMS]
    return " OR ".join(chosen) if chosen else None


def source_filter(source_ids: list[str]) -> str:
    """FTS5 column filter narrowing a match to these sources before any row is joined.

    A source id is tokenized like text, so `remote-box` also matches `remote-box-old`;
    callers must still check the exact source on the joined rows. This only makes the
    match set small, which is what keeps a common term on a selective source fast.
    """
    phrases = []
    for sid in source_ids:
        parts = _PART.findall(sid.lower())
        if parts:
            phrases.append('"' + " ".join(parts) + '"')
    return "source : (" + " OR ".join(phrases) + ")" if phrases else ""
