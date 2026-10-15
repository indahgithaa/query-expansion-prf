from __future__ import annotations

from src.data import create_split, normalize_filename, validate_split


def _synthetic_qa():
    types = (
        "skrining", "status_gizi", "biokimia", "pes", "tujuan_diet",
        "prinsip_diet", "syarat_diet", "pedoman",
    )
    rows = []
    for doc_index in range(120):
        for kind in types:
            rows.append({"qid": f"Q{doc_index:03d}-{kind}", "doc_id": f"D{doc_index:04d}", "question_type": kind})
    return rows


def test_filename_normalization_handles_hash_unicode_and_word_joiner():
    assert normalize_filename("MECKEL#U2019S\u2060.chunks.json") == "MECKEL’S.chunks.json"


def test_document_split_is_disjoint_complete_and_deterministic():
    qa = _synthetic_qa()
    first = create_split(qa, seed=7)
    second = create_split(qa, seed=7)
    assert first == second
    validate_split(first, qa)
    assert {part: len(docs) for part, docs in first["documents"].items()} == {
        "collection": 72, "dev": 24, "test": 24
    }
