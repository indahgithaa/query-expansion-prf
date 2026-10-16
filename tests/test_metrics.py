import math

import numpy as np

from src.metrics import (
    average_precision,
    binary_ndcg_at_k,
    holm_adjust,
    rank_biserial_effect,
    recall_at_k,
    reciprocal_rank,
    robustness_counts,
    single_gold_ndcg,
)


def test_strict_metrics_match_manual_example():
    assert reciprocal_rank(4) == 0.25
    assert recall_at_k(4, 3) == 0.0
    assert recall_at_k(4, 5) == 1.0
    assert math.isclose(single_gold_ndcg(4, 10), 1.0 / math.log2(5))
    assert single_gold_ndcg(11, 10) == 0.0


def test_multi_gold_metrics_match_manual_example():
    ranking = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert average_precision(ranking, relevant) == 0.5
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    assert math.isclose(binary_ndcg_at_k(ranking, relevant, 3), expected_dcg / expected_idcg)


def test_unreturned_multi_gold_items_contribute_zero_to_ap():
    assert average_precision(["a", "b"], {"b", "missing"}) == 0.25


def test_robustness_counts_match_manual_example():
    result = robustness_counts([1, 4, 3, 7], [2, 4, 5, 6])
    assert result == {"wins": 2, "ties": 1, "losses": 1, "n": 4, "robustness_index": 0.25}


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.03, 0.04])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_rank_biserial_effect_uses_signed_rank_sums():
    assert rank_biserial_effect([1.0, 2.0, -3.0]) == 0.0
    assert rank_biserial_effect([0.0, 0.0]) == 0.0
