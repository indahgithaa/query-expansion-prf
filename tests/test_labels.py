from __future__ import annotations

from src.labels import answer_token_set, build_qrels, overlap_ratio


def test_answer_tokens_remove_protocol_stopwords():
    assert answer_token_set(["Apa tujuan diet pasien ini adalah energi cukup"]) == {
        "tujuan", "diet", "energi"
    }


def test_overlap_ratio_uses_unique_answer_tokens():
    assert overlap_ratio(
        {"energi", "protein", "karbohidrat", "lemak"},
        "Energi, protein, dan karbohidrat",
    ) == 0.75


def test_strict_gold_is_forced_into_multi_gold():
    qa = [{
        "qid": "Q1", "doc_id": "D1", "gold_chunk_id": "C2",
        "answer_texts": ["energi protein karbohidrat"],
    }]
    chunks = [
        {"chunk_id": "C1", "doc_id": "D1", "text": "energi protein karbohidrat"},
        {"chunk_id": "C2", "doc_id": "D1", "text": "teks lain"},
    ]
    strict, multi, audit = build_qrels(qa, chunks)
    assert strict == [{"qid": "Q1", "chunk_id": "C2", "relevance": 1}]
    assert {row["chunk_id"] for row in multi[0.7]} == {"C1", "C2"}
    assert audit["multi_gold"]["strict_gold_forced_relevant"] is True


def test_short_answer_is_excluded_from_multi_gold():
    qa = [{"qid": "Q1", "doc_id": "D1", "gold_chunk_id": "C1", "answer_texts": ["energi cukup"]}]
    chunks = [{"chunk_id": "C1", "doc_id": "D1", "text": "energi cukup"}]
    _, multi, audit = build_qrels(qa, chunks)
    assert all(not rows for rows in multi.values())
    assert audit["multi_gold"]["excluded_qa"] == 1
