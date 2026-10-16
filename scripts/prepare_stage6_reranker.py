"""Prepare deduplicated private reranker pairs for all Stage 6/7 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage6_selection import Resources, bm25_pairs, build_features
from src.data import write_json
from src.methods import descending_indices, minmax, sigmoid_weight
from src.reranker import pair_id
from src.text import tokenize


KS = (0.5, 1.0, 2.0, 4.0, 8.0)
TOP_K = 20
MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FALLBACK_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
FALLBACK_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_pair(pairs: dict[str, dict], question: str, passage: str) -> str:
    identifier = pair_id(question, passage)
    row = {"pair_id": identifier, "question": question, "passage": passage}
    if identifier in pairs and pairs[identifier] != row:
        raise RuntimeError("Tabrakan SHA-256 pada pasangan reranker")
    pairs[identifier] = row
    return identifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--output", type=Path, default=Path("kaggle_io/stage6_reranker"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("configs/stage6_pre_reranker.yaml").read_text(encoding="utf-8"))
    m, repeat = int(config["m"]), int(config["r"])
    selected_k, selected_tau = float(config["k"]), float(config["tau"])
    resources = Resources(args.processed, args.embeddings)
    pairs: dict[str, dict] = {}
    audit, total_occurrences = [], 0

    for seed in range(1, 11):
        dev = build_features(resources, args.processed, [seed], "QS-BM25", partition="dev")
        test = build_features(resources, args.processed, [seed], "QS-BM25", partition="test")
        taus = sorted(set(float(value) for value in np.percentile(
            [feature["s"] for feature in dev], np.arange(5, 100, 5),
        )))
        combinations = [(k, tau) for k in KS for tau in taus]
        if (selected_k, selected_tau) not in combinations:
            combinations.append((selected_k, selected_tau))
        seed_before = len(pairs)
        for partition, features in (("dev", dev), ("test", test)):
            bm25 = bm25_pairs(features, m, repeat)
            for feature, (qa_raw, ext_raw) in zip(features, bm25):
                original_question = resources.qa_by_qid[feature["qid"]]["question"]
                b1 = feature["patient"].model.scores(
                    tokenize(feature["query"], remove_stopwords=True),
                )
                candidate_indices = set(descending_indices(b1)[:TOP_K].tolist())
                qa_score, ext_score = minmax(qa_raw), minmax(ext_raw)
                for k, tau in combinations:
                    weight = sigmoid_weight(feature["s"], k, tau)
                    soft = weight * qa_score + (1.0 - weight) * ext_score
                    candidate_indices.update(descending_indices(soft)[:TOP_K].tolist())
                for index in sorted(candidate_indices):
                    add_pair(
                        pairs, original_question,
                        feature["patient"].records[int(index)]["text"],
                    )
                total_occurrences += len(candidate_indices)
        audit.append({
            "seed": seed, "dev_queries": len(dev), "test_queries": len(test),
            "tau_candidates": len(taus), "new_unique_pairs": len(pairs) - seed_before,
        })
        print(json.dumps(audit[-1]), flush=True)

    input_path = args.output / "stage6_reranker_input.jsonl"
    with input_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in pairs.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "fallback_model_id": FALLBACK_ID, "fallback_revision": FALLBACK_REVISION,
        "top_k": TOP_K, "seeds": list(range(1, 11)), "partitions": ["dev", "test"],
        "scenarios": ["TPL", "NEW", "PARA", "PNEW"],
        "unique_pairs": len(pairs), "candidate_occurrences_after_query_union": total_occurrences,
        "input_sha256": sha256(input_path), "audit": audit,
        "selection_note": "test pairs are cached only; component selection remains dev-only",
    }
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    archive = args.output / "stage6-reranker-private.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        handle.write(input_path, input_path.name)
        handle.write(manifest_path, manifest_path.name)
    print(json.dumps({**manifest, "archive": str(archive), "archive_sha256": sha256(archive)}, indent=2))


if __name__ == "__main__":
    main()
