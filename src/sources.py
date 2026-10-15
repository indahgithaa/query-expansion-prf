"""Patient retrieval indexes, feedback-source indexes, and leakage checks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.bm25 import BM25, seeded_tie_rank, stable_rank
from src.text import tokenize


class PatientBM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("Indeks pasien kosong")
        self.records = sorted(chunks, key=lambda row: row["chunk_id"])
        self.item_ids = [row["chunk_id"] for row in self.records]
        self.model = BM25([tokenize(row["text"]) for row in self.records])

    def search(self, query: str, *, remove_stopwords: bool, top_k: int | None = None):
        ranking = stable_rank(
            self.model.scores(tokenize(query, remove_stopwords=remove_stopwords)),
            self.item_ids,
            top_k=top_k,
        )
        return [{**self.records[item.index], **asdict(item)} for item in ranking]


class QuestionFeedbackIndex:
    def __init__(self, qa: list[dict[str, Any]], tie_seed: int = 1729) -> None:
        if not qa:
            raise ValueError("Indeks pertanyaan kosong")
        self.records = sorted(qa, key=lambda row: row["qid"])
        self.item_ids = [row["qid"] for row in self.records]
        self.doc_ids = {row["doc_id"] for row in self.records}
        self.tie_seed = tie_seed
        self.model = BM25([tokenize(row["question"]) for row in self.records])

    def search(self, query: str, *, query_id: str, top_k: int = 3):
        ranking = seeded_tie_rank(
            self.model.scores(tokenize(query)), self.item_ids,
            seed=self.tie_seed, query_id=query_id, top_k=top_k,
        )
        return [{**self.records[item.index], **asdict(item)} for item in ranking]


class ExternalFeedbackIndex:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("Indeks chunk eksternal kosong")
        self.records = sorted(chunks, key=lambda row: row["chunk_id"])
        self.item_ids = [row["chunk_id"] for row in self.records]
        self.doc_ids = {row["doc_id"] for row in self.records}
        self.model = BM25([tokenize(row["text"]) for row in self.records])

    def search(self, query: str, *, target_doc_id: str, top_k: int = 3):
        if target_doc_id in self.doc_ids:
            raise ValueError("Dokumen target berada dalam koleksi feedback")
        ranking = stable_rank(
            self.model.scores(tokenize(query, remove_stopwords=True)),
            self.item_ids,
            top_k=top_k,
        )
        return [{**self.records[item.index], **asdict(item)} for item in ranking]


def select_feedback_records(qa, chunks, split):
    collection_docs = set(split["documents"]["collection"])
    collection_qids = set(split["qids"]["collection"])
    selected_qa = [row for row in qa if row["doc_id"] in collection_docs and row["qid"] in collection_qids]
    selected_chunks = [row for row in chunks if row["doc_id"] in collection_docs]
    return selected_qa, selected_chunks


def validate_no_feedback_leakage(qa_collection, chunk_collection, split):
    collection = set(split["documents"]["collection"])
    dev = set(split["documents"]["dev"])
    test = set(split["documents"]["test"])
    qa_docs = {row["doc_id"] for row in qa_collection}
    chunk_docs = {row["doc_id"] for row in chunk_collection}
    expected_qids = set(split["qids"]["collection"])
    actual_qids = {row["qid"] for row in qa_collection}
    failures = {
        "qa_outside_collection": sorted(qa_docs - collection),
        "chunks_outside_collection": sorted(chunk_docs - collection),
        "qa_dev_overlap": sorted(qa_docs & dev),
        "qa_test_overlap": sorted(qa_docs & test),
        "chunk_dev_overlap": sorted(chunk_docs & dev),
        "chunk_test_overlap": sorted(chunk_docs & test),
        "missing_qids": sorted(expected_qids - actual_qids),
        "extra_qids": sorted(actual_qids - expected_qids),
    }
    if any(failures.values()):
        raise ValueError(f"Kebocoran sumber feedback: {failures}")
    return {
        "status": "PASS",
        "collection_documents": len(collection),
        "question_records": len(qa_collection),
        "external_chunks": len(chunk_collection),
        "dev_document_overlap": 0,
        "test_document_overlap": 0,
    }
