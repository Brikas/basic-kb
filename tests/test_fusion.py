"""Reciprocal rank fusion. Cases follow qdrant-client hybrid/test_reranking.py and qmd's
RRF tests (see SPEC.md), with ranks counted from 1 as in Cormack et al."""
from __future__ import annotations

import pytest

from basic_kb.fusion import rrf


def test_a_key_in_both_lists_beats_either_lists_top():
    fused = dict(rrf([["1", "2", "3"], ["5", "6", "7", "2"]], k=1))
    assert max(fused, key=fused.get) == "2"
    assert fused["2"] == pytest.approx(1 / 3 + 1 / 5)
    assert fused["1"] == pytest.approx(1 / 2) == fused["5"]


def test_empty_lists():
    assert rrf([]) == []
    assert rrf([[]]) == []
    assert rrf([["1", "2", "3"], []], k=1) == [("1", pytest.approx(1 / 2)), ("2", pytest.approx(1 / 3)),
                                              ("3", pytest.approx(1 / 4))]


def test_a_single_list_keeps_its_order():
    assert [key for key, _ in rrf([list("abcde")], k=20)] == list("abcde")


def test_weights_multiply_and_zero_drops_a_list():
    fused = dict(rrf([["a"], ["b"]], k=0, weights=[2.0, 1.0]))
    assert fused == {"a": pytest.approx(2.0), "b": pytest.approx(1.0)}
    assert dict(rrf([["a"], ["b"]], k=0, weights=[1.0, 0.0])) == {"a": pytest.approx(1.0)}


def test_bad_arguments_fail_loudly():
    with pytest.raises(ValueError, match="weights"):
        rrf([["a"], ["b"]], weights=[1.0])
    with pytest.raises(ValueError, match="k must be"):
        rrf([["a"]], k=-1)


def test_a_repeat_inside_one_list_counts_once_at_its_best_rank():
    assert dict(rrf([["a", "b", "a"]], k=0)) == {"a": pytest.approx(1.0), "b": pytest.approx(0.5)}


def test_equal_scores_order_by_first_appearance():
    assert [key for key, _ in rrf([["x"], ["y"]])] == ["x", "y"]
    assert [key for key, _ in rrf([["y"], ["x"]])] == ["y", "x"]


def test_equal_scores_order_by_best_rank_first():
    # At k=0, p (rank 2 in two lists) = 1/2 + 1/2 and q (rank 1 in one) = 1: equal. q ranked higher.
    fused = rrf([["z", "p"], ["w", "p"], ["q"]], k=0)
    assert dict(fused)["p"] == pytest.approx(1.0) == dict(fused)["q"]
    order = [key for key, _ in fused]
    assert order.index("q") < order.index("p")


def test_k_trades_a_top_position_against_agreement_further_down():
    # "solo" tops one list; "both" is third in two. Small k: the top position wins.
    # Large k: credit decays slowly, so agreement wins.
    lists = [["solo", "a", "both"], ["b", "c", "both"]]
    assert [key for key, _ in rrf(lists, k=0)][0] == "solo"      # 1   > 1/3 + 1/3
    assert [key for key, _ in rrf(lists, k=60)][0] == "both"     # 1/61 < 2/63
