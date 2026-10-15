"""Paraphrase assignment and leakage-safe evaluation scenarios."""

from __future__ import annotations

import hashlib
from typing import Any


SCENARIOS = ("TPL", "NEW", "PARA", "PNEW")


def deterministic_paraphrase(qid: str, seed: int, variants: list[str]) -> str:
    if not variants:
        raise ValueError("Daftar parafrasa kosong")
    digest = hashlib.sha256(f"{seed}|{qid}".encode()).digest()
    return variants[int.from_bytes(digest[:8], "big") % len(variants)]


def scenario_query(row: dict[str, Any], scenario: str, seed: int, paraphrases: dict) -> str:
    if scenario not in SCENARIOS:
        raise ValueError(f"Skenario tidak dikenal: {scenario}")
    if scenario in {"TPL", "NEW"}:
        return row["question"]
    split = "dev" if row["partition"] == "dev" else "test"
    return deterministic_paraphrase(row["qid"], seed, paraphrases[row["question_type"]][split])


def scenario_collection(
    collection_qa: list[dict[str, Any]], question_type: str, scenario: str
) -> list[dict[str, Any]]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Skenario tidak dikenal: {scenario}")
    if scenario in {"NEW", "PNEW"}:
        result = [row for row in collection_qa if row["question_type"] != question_type]
        if any(row["question_type"] == question_type for row in result):
            raise AssertionError("Leave-one-type-out gagal")
        return result
    return list(collection_qa)
