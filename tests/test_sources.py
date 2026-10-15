from __future__ import annotations

import pytest

from src.sources import validate_no_feedback_leakage


def test_feedback_leakage_validator_accepts_collection_only():
    split = {
        "documents": {"collection": ["D1"], "dev": ["D2"], "test": ["D3"]},
        "qids": {"collection": ["Q1"], "dev": ["Q2"], "test": ["Q3"]},
    }
    audit = validate_no_feedback_leakage(
        [{"qid": "Q1", "doc_id": "D1"}], [{"chunk_id": "C1", "doc_id": "D1"}], split
    )
    assert audit["status"] == "PASS"


def test_feedback_leakage_validator_rejects_dev_document():
    split = {
        "documents": {"collection": ["D1"], "dev": ["D2"], "test": ["D3"]},
        "qids": {"collection": ["Q1"], "dev": ["Q2"], "test": ["Q3"]},
    }
    with pytest.raises(ValueError, match="Kebocoran"):
        validate_no_feedback_leakage(
            [{"qid": "Q1", "doc_id": "D2"}], [{"chunk_id": "C1", "doc_id": "D1"}], split
        )
