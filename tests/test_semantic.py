import numpy as np
import pytest

from src.methods import rerank_top_k
from src.semantic import (
    TfidfSimilarity,
    cosine_scores,
    l2_normalize,
    normalized_average,
    rocchio_vector,
    rrf_pair,
)


def test_l2_normalize_and_cosine_scores():
    query = l2_normalize([3.0, 4.0])
    assert np.allclose(query, [0.6, 0.8])
    documents = np.asarray([[0.6, 0.8], [0.8, -0.6]], dtype=np.float32)
    assert np.allclose(cosine_scores(query, documents), [1.0, 0.0], atol=1e-7)


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0])


def test_rocchio_vector_matches_manual_result():
    result = rocchio_vector([1.0, 0.0], np.asarray([[0.0, 1.0], [0.0, 1.0]]), alpha=1, beta=1)
    assert np.allclose(result, [2 ** -0.5, 2 ** -0.5])


def test_normalized_average_and_rrf_pair():
    assert np.allclose(normalized_average([0, 2, 1], [4, 0, 2]), [0.5, 0.5, 0.5])
    fused = rrf_pair([3, 2, 1], [1, 3, 2])
    assert int(np.argmax(fused)) == 1


def test_word_tfidf_prefers_matching_document():
    index = TfidfSimilarity(["diet rendah garam", "skrining status gizi"], analyzer="word")
    scores = index.scores("status gizi")
    assert scores[1] > scores[0]


def test_character_tfidf_handles_surface_variation():
    index = TfidfSimilarity(["biokimia pasien", "tujuan diet"], analyzer="char_3_5")
    scores = index.scores("data biokimia")
    assert scores[0] > scores[1]


def test_rerank_top_k_keeps_remainder_below_candidates():
    pipeline = np.asarray([0.9, 0.8, 0.7, 0.6])
    order = rerank_top_k(pipeline, [0, 1], [0.0, 1.0], weight=1.0)
    assert order.tolist() == [1, 0, 2, 3]
    assert rerank_top_k(pipeline, [0, 1], [0.0, 1.0], weight=0.0).tolist() == [0, 1, 2, 3]
