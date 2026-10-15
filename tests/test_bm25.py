from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from src.bm25 import BM25, seeded_tie_rank, stable_rank


def test_scores_match_rank_bm25():
    corpus = [["diet", "energi"], ["diet", "protein", "protein"], ["laboratorium", "darah"]]
    query = ["diet", "protein"]
    expected = BM25Okapi(corpus, k1=1.5, b=0.75).get_scores(query)
    actual = BM25(corpus, k1=1.5, b=0.75).scores(query)
    assert np.allclose(actual, expected, atol=1e-12)


def test_stable_rank_keeps_input_order_for_ties():
    ranked = stable_rank([1.0, 1.0, 0.0], ["C2", "C1", "C3"])
    assert [item.item_id for item in ranked] == ["C2", "C1", "C3"]


def test_seeded_tie_rank_is_reproducible_but_not_optimistic():
    ids = [f"Q{i}" for i in range(20)]
    first = seeded_tie_rank([0.0] * 20, ids, seed=17, query_id="target")
    second = seeded_tie_rank([0.0] * 20, ids, seed=17, query_id="target")
    assert first == second
    assert [item.item_id for item in first] != ids
    assert [item.rank for item in first] == list(range(1, 21))
