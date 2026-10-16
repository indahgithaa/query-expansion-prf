"""Evaluation metrics and paired statistical helpers for retrieval runs."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
from scipy.stats import rankdata


def reciprocal_rank(rank: int) -> float:
    """Return reciprocal rank for a one-based rank."""

    if rank < 1:
        raise ValueError("rank must be at least one")
    return 1.0 / rank


def recall_at_k(rank: int, k: int) -> float:
    """Return single-gold recall at cutoff *k*."""

    if rank < 1 or k < 1:
        raise ValueError("rank and k must be at least one")
    return float(rank <= k)


def single_gold_ndcg(rank: int, k: int = 10) -> float:
    """Return nDCG@k for a ranking with exactly one relevant item."""

    if rank < 1 or k < 1:
        raise ValueError("rank and k must be at least one")
    if rank > k:
        return 0.0
    return float(1.0 / np.log2(rank + 1.0))


def average_precision(ranking: Sequence[str], relevant: Collection[str]) -> float:
    """Return AP over the supplied ranking; unreturned relevant items score zero."""

    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant must not be empty")
    hits = 0
    precision_sum = 0.0
    for position, item_id in enumerate(ranking, 1):
        if item_id in relevant_set:
            hits += 1
            precision_sum += hits / position
    return precision_sum / len(relevant_set)


def binary_ndcg_at_k(
    ranking: Sequence[str], relevant: Collection[str], k: int = 10
) -> float:
    """Return binary-gain nDCG@k."""

    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant must not be empty")
    if k < 1:
        raise ValueError("k must be at least one")
    dcg = sum(
        1.0 / np.log2(position + 1.0)
        for position, item_id in enumerate(ranking[:k], 1)
        if item_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(position + 1.0) for position in range(1, ideal_hits + 1))
    return float(dcg / idcg)


def robustness_counts(ranks: Sequence[int], baseline_ranks: Sequence[int]) -> dict[str, float | int]:
    """Count rank wins/ties/losses and compute (wins-losses)/N."""

    values = np.asarray(ranks, dtype=int)
    baseline = np.asarray(baseline_ranks, dtype=int)
    if values.shape != baseline.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("ranks and baseline_ranks must be non-empty aligned vectors")
    wins = int(np.sum(values < baseline))
    ties = int(np.sum(values == baseline))
    losses = int(np.sum(values > baseline))
    total = int(values.size)
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "n": total,
        "robustness_index": (wins - losses) / total,
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a finite one-dimensional vector in [0, 1]")
    count = len(values)
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(count, dtype=float)
    running = 0.0
    for index, original_index in enumerate(order):
        candidate = (count - index) * values[original_index]
        running = max(running, candidate)
        adjusted_sorted[index] = min(running, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def rank_biserial_effect(differences: Sequence[float]) -> float:
    """Return matched-pairs rank-biserial correlation, positive for improvement."""

    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)
