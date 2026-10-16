"""Semantic ranking helpers used by Stage 6 component selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.methods import descending_indices, minmax, rrf_scores


def l2_normalize(vector: Sequence[float]) -> np.ndarray:
    """Return a finite unit vector and reject degenerate inputs."""
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Vektor harus satu dimensi dan finite")
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("Vektor nol tidak dapat dinormalisasi")
    return value / norm


def cosine_scores(query_vector: Sequence[float], document_vectors: np.ndarray) -> np.ndarray:
    """Cosine scores for already-normalized document vectors."""
    query = l2_normalize(query_vector)
    documents = np.asarray(document_vectors, dtype=np.float32)
    if documents.ndim != 2 or documents.shape[1] != query.shape[0]:
        raise ValueError("Dimensi embedding tidak cocok")
    if not np.isfinite(documents).all():
        raise ValueError("Embedding dokumen tidak finite")
    return documents @ query


def rocchio_vector(
    query_vector: Sequence[float], feedback_vectors: np.ndarray, *, alpha: float, beta: float,
) -> np.ndarray:
    """Build a normalized positive-feedback Rocchio query vector."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha dan beta Rocchio harus positif")
    feedback = np.asarray(feedback_vectors, dtype=np.float32)
    if feedback.ndim != 2 or feedback.shape[0] == 0:
        raise ValueError("Feedback Rocchio harus berupa matriks tidak kosong")
    return l2_normalize(alpha * l2_normalize(query_vector) + beta * feedback.mean(axis=0))


def normalized_average(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    """Average two independently min-max-normalized score vectors."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError("Vektor skor harus memiliki bentuk yang sama")
    return 0.5 * (minmax(first_array) + minmax(second_array))


def rrf_pair(first: Sequence[float], second: Sequence[float], *, k: int = 60) -> np.ndarray:
    """Fuse two score vectors using stable descending rankings."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError("Vektor skor harus memiliki bentuk yang sama")
    return rrf_scores(
        [descending_indices(first_array), descending_indices(second_array)], len(first_array), k=k,
    )
