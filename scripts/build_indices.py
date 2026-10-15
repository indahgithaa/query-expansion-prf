"""Build and verify Stage 3 sparse retrieval/feedback indexes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rank_bm25 import BM25Okapi

from src.bm25 import BM25
from src.data import read_json, read_jsonl, write_json
from src.sources import (
    ExternalFeedbackIndex,
    PatientBM25Index,
    QuestionFeedbackIndex,
    select_feedback_records,
    validate_no_feedback_leakage,
)
from src.text import tokenize


def dump_pickle(path: Path, value) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/indices"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/indices"))
    args = parser.parse_args()

    qa = read_jsonl(args.processed / "qa.jsonl")
    chunks = read_jsonl(args.processed / "chunks.jsonl")
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc_id"]].append(chunk)

    patient_indexes = {doc_id: PatientBM25Index(rows) for doc_id, rows in sorted(by_doc.items())}
    files = {"patient_indexes": dump_pickle(args.models_dir / "patient_bm25.pkl", patient_indexes)}

    leakage_audits = []
    source_counts = []
    for seed in range(1, 11):
        split = read_json(args.processed / "splits" / f"seed_{seed:02d}.json")
        collection_qa, collection_chunks = select_feedback_records(qa, chunks, split)
        leakage = validate_no_feedback_leakage(collection_qa, collection_chunks, split)
        leakage_audits.append({"seed": seed, **leakage})
        bundle = {
            "seed": seed,
            "question": QuestionFeedbackIndex(collection_qa, tie_seed=seed * 1009),
            "external": ExternalFeedbackIndex(collection_chunks),
        }
        files[f"seed_{seed:02d}"] = dump_pickle(args.models_dir / f"feedback_seed_{seed:02d}.pkl", bundle)
        source_counts.append(
            {"seed": seed, "question_records": len(collection_qa), "external_chunks": len(collection_chunks)}
        )
        del bundle
        gc.collect()

    sample_corpus = [
        ["diet", "energi"],
        ["diet", "protein", "protein"],
        ["laboratorium", "darah"],
    ]
    sample_query = ["diet", "protein"]
    custom_scores = BM25(sample_corpus).scores(sample_query)
    reference_scores = BM25Okapi(sample_corpus).get_scores(sample_query)
    maximum_difference = float(abs(custom_scores - reference_scores).max())
    if maximum_difference > 1e-12:
        raise AssertionError(f"BM25 berbeda dari rank_bm25: {maximum_difference}")

    audit = {
        "patient_indexes": {"documents": len(patient_indexes), "chunks": len(chunks)},
        "feedback_sources": source_counts,
        "leakage": leakage_audits,
        "bm25_reference": {
            "package": "rank_bm25.BM25Okapi",
            "maximum_absolute_score_difference": maximum_difference,
            "status": "PASS",
        },
        "ranking_ties": {
            "patient_and_external": "stable_original_chunk_index_order",
            "question_similarity": "query_specific_seeded_random",
            "optimistic_ranking": False,
        },
        "index_files": files,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_dir / "audit.json", audit)
    write_json(args.models_dir / "manifest.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
