"""split_text: recursive character splitting shared by both chunkers."""
from __future__ import annotations

from basic_kb.textsplit import split_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_gives_no_chunks():
    assert split_text("", 100, 10, 5) == []
    assert split_text("   \n\n  ", 100, 10, 5) == []


def test_short_text_is_one_chunk():
    assert split_text("hello world", 100, 10, 5) == ["hello world"]


def test_every_chunk_fits_chunk_size():
    text = words(300)
    for c in split_text(text, 50, 10, 5):
        assert len(c) <= 50


def test_all_words_survive_splitting():
    text = words(300)
    joined = " ".join(split_text(text, 50, 0, 1))
    assert set(joined.split()) == set(text.split())


def test_trailing_piece_below_min_chunk_is_dropped():
    text = words(300)                       # ends in "w299", 4 chars
    joined = " ".join(split_text(text, 50, 0, 5))
    assert "w299" not in joined.split()


def test_overlap_carries_the_tail_into_the_next_chunk():
    chunks = split_text(words(200), 50, 10, 5)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:]):
        aw, bw = a.split(" "), b.split(" ")
        # b opens with some non-empty suffix of a's words.
        assert any(aw[-k:] == bw[:k] for k in range(1, len(aw) + 1))


def test_zero_overlap_repeats_nothing():
    chunks = split_text(words(200), 50, 0, 5)
    seen: set[str] = set()
    for c in chunks:
        ws = set(c.split())
        assert not (ws & seen)
        seen |= ws


def test_chunks_below_min_chunk_are_dropped():
    # Two paragraphs that cannot merge into one chunk; the second is below min_chunk.
    text = "This paragraph is long enough to keep around.\n\nno"
    out = split_text(text, 46, 0, 10)
    assert out == ["This paragraph is long enough to keep around."]


def test_small_piece_merges_when_it_fits():
    text = "This paragraph is long enough to keep around.\n\nno"
    assert split_text(text, 60, 0, 10) == [text]


def test_paragraph_separator_wins_when_present():
    a = "A" * 30
    b = "B" * 30
    assert split_text(f"{a}\n\n{b}", 40, 0, 5) == [a, b]


def test_oversized_piece_recurses_to_finer_separators():
    # One "paragraph" with no blank lines but many sentences.
    text = ". ".join(f"Sentence number {i} is here" for i in range(40)) + "."
    out = split_text(text, 80, 0, 5)
    assert len(out) > 1
    assert all(len(c) <= 80 for c in out)


def test_custom_separators_are_honoured():
    assert split_text("a|b|c", 1, 0, 1, separators=["|"]) == ["a", "b", "c"]
