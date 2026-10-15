"""Strict and token-overlap multi-gold relevance labels."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable

from src.text import tokenize


def answer_token_set(answer_texts: Iterable[str]) -> set[str]:
    return set(tokenize(" ".join(answer_texts), remove_stopwords=True))


def overlap_ratio(answer_tokens: set[str], chunk_text: str) -> float:
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & set(tokenize(chunk_text, remove_stopwords=True))) / len(answer_tokens)


def build_qrels(
    qa: list[dict[str, Any]], chunks: list[dict[str, Any]], thresholds: tuple[float, ...] = (0.5, 0.6, 0.7)
) -> tuple[list[dict[str, Any]], dict[float, list[dict[str, Any]]], dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc_id"]].append(chunk)

    strict = [{"qid": row["qid"], "chunk_id": row["gold_chunk_id"], "relevance": 1} for row in qa]
    multi = {threshold: [] for threshold in thresholds}
    eligible = 0
    excluded = []

    for row in qa:
        answer_tokens = answer_token_set(row["answer_texts"])
        if len(answer_tokens) < 3:
            excluded.append({"qid": row["qid"], "answer_token_count": len(answer_tokens)})
            continue
        eligible += 1
        ratios = {
            chunk["chunk_id"]: overlap_ratio(answer_tokens, chunk["text"])
            for chunk in by_doc[row["doc_id"]]
        }
        for threshold in thresholds:
            relevant = {chunk_id for chunk_id, ratio in ratios.items() if ratio >= threshold}
            relevant.add(row["gold_chunk_id"])
            multi[threshold].extend(
                {"qid": row["qid"], "chunk_id": chunk_id, "relevance": 1}
                for chunk_id in sorted(relevant)
            )

    sensitivity = {}
    for threshold, rows in multi.items():
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["qid"]] += 1
        values = [counts[row["qid"]] for row in qa if row["qid"] in counts]
        sensitivity[f"{threshold:.1f}"] = {
            "eligible_qa": len(values),
            "qrels": len(rows),
            "mean_relevant_chunks": mean(values) if values else 0.0,
            "median_relevant_chunks": median(values) if values else 0.0,
            "min_relevant_chunks": min(values) if values else 0,
            "max_relevant_chunks": max(values) if values else 0,
            "qa_with_multiple_relevant_chunks": sum(value > 1 for value in values),
        }
    audit = {
        "strict": {"qa": len(qa), "qrels": len(strict), "exactly_one_per_qa": True},
        "multi_gold": {
            "answer_token_rule": "lowercase_regex_w_without_stopwords_unique_set",
            "minimum_answer_tokens": 3,
            "eligible_qa": eligible,
            "excluded_qa": len(excluded),
            "excluded": excluded,
            "strict_gold_forced_relevant": True,
            "sensitivity": sensitivity,
        },
    }
    return strict, multi, audit
