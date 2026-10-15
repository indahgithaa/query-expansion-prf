"""Auditable Okapi BM25 with explicit, deterministic tie policies."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RankedItem:
    rank: int
    item_id: str
    score: float
    index: int


class BM25:
    """Okapi BM25 compatible with rank_bm25.BM25Okapi defaults."""

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if not corpus:
            raise ValueError("Corpus BM25 kosong")
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)
        self.corpus_size = len(corpus)
        self.doc_freqs = [Counter(document) for document in corpus]
        self.doc_len = np.asarray([len(document) for document in corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean())
        if self.avgdl <= 0:
            raise ValueError("Semua dokumen BM25 kosong")

        df: Counter[str] = Counter()
        for frequencies in self.doc_freqs:
            df.update(frequencies.keys())
        raw_idf = {
            term: math.log(self.corpus_size - frequency + 0.5) - math.log(frequency + 0.5)
            for term, frequency in df.items()
        }
        self.average_idf = sum(raw_idf.values()) / len(raw_idf)
        floor = self.epsilon * self.average_idf
        self.idf = {term: floor if value < 0 else value for term, value in raw_idf.items()}

    def term_scores(self, term: str) -> np.ndarray:
        frequency = np.asarray([doc.get(term, 0) for doc in self.doc_freqs], dtype=float)
        denominator = frequency + self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        return self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator

    def scores(self, query_tokens: Iterable[str]) -> np.ndarray:
        values = np.zeros(self.corpus_size, dtype=float)
        for term in query_tokens:
            values += self.term_scores(term)
        return values

    def weighted_scores(self, weights: dict[str, float]) -> np.ndarray:
        values = np.zeros(self.corpus_size, dtype=float)
        for term, weight in weights.items():
            if weight < 0 or not math.isfinite(float(weight)):
                raise ValueError("Bobot query harus finite dan nonnegatif")
            values += float(weight) * self.term_scores(term)
        return values

    def explain(self, query_tokens: Iterable[str], document_index: int) -> dict[str, float]:
        if not 0 <= document_index < self.corpus_size:
            raise IndexError(document_index)
        result: Counter[str] = Counter()
        for term in query_tokens:
            result[term] += float(self.term_scores(term)[document_index])
        return dict(result)


def _validate_rank_inputs(scores: Sequence[float], item_ids: Sequence[str]) -> None:
    if len(scores) != len(item_ids):
        raise ValueError("Panjang scores dan item_ids berbeda")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids tidak unik")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("Score tidak finite")


def stable_rank(
    scores: Sequence[float], item_ids: Sequence[str], *, top_k: int | None = None
) -> list[RankedItem]:
    """Descending score; exact ties retain the input/chunk index order."""
    _validate_rank_inputs(scores, item_ids)
    indices = sorted(range(len(item_ids)), key=lambda index: (-float(scores[index]), index))
    if top_k is not None:
        indices = indices[:top_k]
    return [
        RankedItem(rank=rank, item_id=item_ids[index], score=float(scores[index]), index=index)
        for rank, index in enumerate(indices, 1)
    ]


def seeded_tie_rank(
    scores: Sequence[float],
    item_ids: Sequence[str],
    *,
    seed: int,
    query_id: str,
    top_k: int | None = None,
) -> list[RankedItem]:
    """Descending score; exact ties use a query-specific fixed random order."""
    _validate_rank_inputs(scores, item_ids)
    digest = hashlib.sha256(f"{seed}|{query_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    indices = list(range(len(item_ids)))
    rng.shuffle(indices)
    indices.sort(key=lambda index: float(scores[index]), reverse=True)
    if top_k is not None:
        indices = indices[:top_k]
    return [
        RankedItem(rank=rank, item_id=item_ids[index], score=float(scores[index]), index=index)
        for rank, index in enumerate(indices, 1)
    ]
