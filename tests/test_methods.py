import math

import numpy as np

from src.bm25 import seeded_tie_rank, stable_rank
from src.methods import expanded_weights, minmax, rrf_scores, sigmoid_weight, softmax


def test_sigmoid_and_minmax_manual_values():
    assert math.isclose(sigmoid_weight(2.0, 1.0, 2.0), 0.5)
    assert np.allclose(minmax([2, 4, 6]), [0, 0.5, 1])
    assert np.array_equal(minmax([3, 3]), [0, 0])


def test_rrf_manual_values():
    scores = rrf_scores([[0, 1, 2], [1, 0, 2]], size=3, k=60)
    assert math.isclose(scores[0], 1 / 61 + 1 / 62)
    assert math.isclose(scores[1], 1 / 62 + 1 / 61)
    assert scores[2] == 2 / 63


def test_feedback_weights_and_softmax():
    values = expanded_weights(["diet"], [["energi", "energi"], ["protein"]], 3, [0.75, 0.25])
    assert values == {"diet": 3.0, "energi": 1.5, "protein": 0.25}
    assert np.isclose(softmax([0, 0]).sum(), 1.0)


def test_tie_policies_are_reproducible_and_not_optimistic():
    stable = stable_rank([1, 1, 0], ["a", "b", "c"])
    assert [item.item_id for item in stable] == ["a", "b", "c"]
    first = seeded_tie_rank([1, 1, 1], ["a", "b", "c"], seed=4, query_id="q")
    second = seeded_tie_rank([1, 1, 1], ["a", "b", "c"], seed=4, query_id="q")
    assert [item.item_id for item in first] == [item.item_id for item in second]
    assert [item.rank for item in first] == [1, 2, 3]
