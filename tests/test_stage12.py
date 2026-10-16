import math

from scripts.run_stage12 import ndcg10, reciprocal_rank


def test_single_gold_metrics():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(14) == 1 / 14
    assert ndcg10(1) == 1.0
    assert ndcg10(14) == 0.0
    assert math.isclose(ndcg10(3), 0.5)
