"""Core ranking formulas for baselines and question-similarity PRF."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def minmax(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Skor harus berupa vektor finite")
    low, high = float(values.min()), float(values.max())
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def sigmoid_weight(score: float, k: float, tau: float) -> float:
    value = float(k) * (float(score) - float(tau))
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def rrf_scores(rankings: Sequence[Sequence[int]], size: int, k: int = 60) -> np.ndarray:
    scores = np.zeros(size, dtype=float)
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("Ranking RRF mengandung indeks duplikat")
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1.0 / (k + rank)
    return scores


def descending_indices(scores: Sequence[float]) -> np.ndarray:
    """Stable descending order; exact ties retain input order."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def rank_of(scores: Sequence[float], gold_index: int) -> int:
    order = descending_indices(scores)
    return int(np.flatnonzero(order == gold_index)[0]) + 1


def expanded_weights(
    query_tokens: Iterable[str], feedback_token_lists: Sequence[Sequence[str]], repeat: int,
    feedback_weights: Sequence[float] | None = None,
) -> dict[str, float]:
    result: Counter[str] = Counter()
    for token in query_tokens:
        result[token] += float(repeat)
    weights = feedback_weights if feedback_weights is not None else [1.0] * len(feedback_token_lists)
    if len(weights) != len(feedback_token_lists):
        raise ValueError("Jumlah bobot feedback tidak cocok")
    for tokens, weight in zip(feedback_token_lists, weights):
        for token in tokens:
            result[token] += float(weight)
    return dict(result)


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature harus positif")
    values = np.asarray(values, dtype=float) / temperature
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def rm3_query_weights(
    query_tokens: Sequence[str], feedback_documents: Sequence[Sequence[str]],
    feedback_scores: Sequence[float], feedback_terms: int, interpolation: float,
) -> dict[str, float]:
    """A compact relevance-model approximation with score-weighted document LMs."""
    if not 0 <= interpolation <= 1:
        raise ValueError("Interpolasi RM3 harus dalam [0,1]")
    scores = np.asarray(feedback_scores, dtype=float)
    scores = scores - scores.max()
    doc_weights = np.exp(scores)
    doc_weights /= doc_weights.sum()
    relevance: Counter[str] = Counter()
    for tokens, weight in zip(feedback_documents, doc_weights):
        counts = Counter(tokens)
        length = max(1, sum(counts.values()))
        for term, frequency in counts.items():
            relevance[term] += float(weight) * frequency / length
    selected = sorted(relevance, key=lambda term: (-relevance[term], term))[:feedback_terms]
    q_counts = Counter(query_tokens)
    q_total = max(1, sum(q_counts.values()))
    weights = {term: interpolation * count / q_total for term, count in q_counts.items()}
    for term in selected:
        weights[term] = weights.get(term, 0.0) + (1.0 - interpolation) * relevance[term]
    return weights
