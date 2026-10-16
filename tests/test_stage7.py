import numpy as np

from scripts.run_stage7 import rank_from_order, scenario_mrr, select_reranker_weight


def test_rank_from_order():
    assert rank_from_order(np.asarray([2, 0, 1]), 0) == 2


def test_scenario_mrr_uses_equal_scenario_weighting():
    rows = [
        {"scenario": "TPL", "rr": 1.0},
        {"scenario": "TPL", "rr": 0.0},
        {"scenario": "NEW", "rr": 0.25},
        {"scenario": "PARA", "rr": 0.5},
        {"scenario": "PNEW", "rr": 0.75},
    ]
    result = scenario_mrr(rows)
    assert result["TPL"] == 0.5
    assert result["mean"] == 0.5


def test_reranker_selection_rejects_best_mean_when_one_scenario_drops_too_far():
    baseline = {
        "lambda": 0.0, "TPL": 0.60, "NEW": 0.30,
        "PARA": 0.30, "PNEW": 0.20, "mean": 0.35,
    }
    eligible = {
        "lambda": 0.25, "TPL": 0.60, "NEW": 0.32,
        "PARA": 0.33, "PNEW": 0.22, "mean": 0.3675,
    }
    rejected = {
        "lambda": 0.5, "TPL": 0.58, "NEW": 0.40,
        "PARA": 0.40, "PNEW": 0.30, "mean": 0.42,
    }
    selected, candidates = select_reranker_weight([baseline, eligible, rejected])
    assert selected["lambda"] == 0.25
    assert [row["lambda"] for row in candidates] == [0.0, 0.25]
