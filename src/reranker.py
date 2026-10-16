"""Utilities for deterministic cross-encoder score caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def pair_id(question: str, passage: str) -> str:
    payload = json.dumps(
        [question, passage], ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_score_cache(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, dict) or not rows:
        raise ValueError("Cache skor reranker kosong atau tidak valid")
    result = {str(key): float(value) for key, value in rows.items()}
    if not all(len(key) == 64 for key in result):
        raise ValueError("ID pasangan reranker tidak valid")
    return result
