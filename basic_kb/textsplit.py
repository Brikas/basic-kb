"""Recursive character text splitter shared by chunkers."""
from __future__ import annotations

from typing import Optional

# Split priority: paragraph → line → sentence → clause → word → char.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", ", ", " ", ""]


def _merge_splits(splits: list[str], separator: str, chunk_size: int, overlap: int, min_chunk: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator)

    for s in splits:
        s_len = len(s)
        join_cost = sep_len if current else 0
        if current_len + join_cost + s_len > chunk_size and current:
            merged = separator.join(current)
            if len(merged) >= min_chunk:
                chunks.append(merged)
            while current and current_len > overlap:
                removed = current.pop(0)
                current_len -= len(removed) + sep_len
            current_len = max(0, current_len)
        current.append(s)
        current_len += s_len + (sep_len if len(current) > 1 else 0)

    if current:
        merged = separator.join(current)
        if len(merged) >= min_chunk:
            chunks.append(merged)
    return chunks


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk: int,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Recursively split text into chunks ≤ chunk_size.
    Tries separators from coarsest to finest; recurses on oversized pieces.
    """
    if separators is None:
        separators = _SEPARATORS

    separator = separators[-1]
    remaining_seps: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_seps = separators[i + 1:]
            break

    raw_parts = text.split(separator) if separator else list(text)
    parts = [p for p in raw_parts if p.strip()]
    good: list[str] = []
    chunks: list[str] = []

    for part in parts:
        if len(part) <= chunk_size:
            good.append(part)
        else:
            if good:
                chunks.extend(_merge_splits(good, separator, chunk_size, overlap, min_chunk))
                good = []
            if remaining_seps:
                chunks.extend(split_text(part, chunk_size, overlap, min_chunk, remaining_seps))
            elif len(part) >= min_chunk:
                chunks.append(part)

    if good:
        chunks.extend(_merge_splits(good, separator, chunk_size, overlap, min_chunk))
    return chunks
