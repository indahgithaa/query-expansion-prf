"""Select reranker weights on dev seeds 1--3 and freeze the FINAL config."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage6_selection import Resources, bm25_pairs, build_features
from src.data import write_json
from src.methods import descending_indices, minmax, rerank_top_k, sigmoid_weight
from src.reranker import load_score_cache, pair_id
from src.text import tokenize


SCENARIOS = ("TPL", "NEW", "PARA", "PNEW")
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(float(row["rr"]))
    result = {scenario: float(np.mean(grouped[scenario])) for scenario in SCENARIOS}
    result["mean"] = float(np.mean(list(result.values())))
    return result


def eligible(candidate: dict, previous: dict) -> bool:
    return all(candidate[scenario] >= previous[scenario] - 0.01 for scenario in SCENARIOS)


def evaluate_grid(
    features: list[dict], pipeline_scores: list[np.ndarray], score_cache: dict[str, float],
    resources: Resources, label: str, top_k: int,
) -> tuple[list[dict], list[dict]]:
    grid_rows: list[dict] = []
    per_query_rows: list[dict] = []
    for weight in LAMBDAS:
        current = []
        for feature, scores in zip(features, pipeline_scores):
            candidates = descending_indices(scores)[:top_k]
            question = resources.qa_by_qid[feature["qid"]]["question"]
            reranker_scores = np.asarray([
                score_cache[pair_id(question, feature["patient"].records[int(index)]["text"])]
                for index in candidates
            ])
            ranking = rerank_top_k(scores, candidates, reranker_scores, weight)
            rank = int(np.flatnonzero(ranking == feature["gold_index"])[0]) + 1
            row = {
                "pipeline": label,
                "lambda": weight,
                "seed": feature["seed"],
                "qid": feature["qid"],
                "scenario": feature["scenario"],
                "rank": rank,
                "rr": 1.0 / rank,
            }
            current.append(row)
            per_query_rows.append(row)
        grid_rows.append({"pipeline": label, "lambda": weight, **summary(current)})
    return grid_rows, per_query_rows


def select_weight(rows: list[dict], *, constrain_against_zero: bool) -> tuple[dict, list[dict]]:
    previous = next(row for row in rows if row["lambda"] == 0.0)
    candidates = [row for row in rows if eligible(row, previous)] if constrain_against_zero else rows
    if not candidates:
        candidates = [previous]
    selected = max(candidates, key=lambda row: (row["mean"], -row["lambda"]))
    return selected, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--score-cache", type=Path, default=Path("models/reranker/stage6_scores.json"))
    parser.add_argument("--manifest", type=Path, default=Path("models/reranker/stage6_manifest.json"))
    parser.add_argument("--results", type=Path, default=Path("results/stage6"))
    args = parser.parse_args()

    pre = yaml.safe_load(Path("configs/stage6_pre_reranker.yaml").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in pre["selection_seeds"]]
    resources = Resources(args.processed, args.embeddings)
    features = build_features(
        resources, args.processed, seeds, pre["question_similarity"], partition="dev",
    )
    bm25 = bm25_pairs(features, int(pre["m"]), int(pre["r"]))
    score_cache = load_score_cache(args.score_cache)
    reranker_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    top_k = int(pre["reranker"]["top_k"])

    b1_scores = [
        feature["patient"].model.scores(tokenize(feature["query"], remove_stopwords=True))
        for feature in features
    ]
    final_scores = []
    for feature, (qa_raw, ext_raw) in zip(features, bm25):
        weight = sigmoid_weight(feature["s"], float(pre["k"]), float(pre["tau"]))
        final_scores.append(weight * minmax(qa_raw) + (1.0 - weight) * minmax(ext_raw))

    b1_grid, b1_queries = evaluate_grid(
        features, b1_scores, score_cache, resources, "B1", top_k,
    )
    final_grid, final_queries = evaluate_grid(
        features, final_scores, score_cache, resources, "FINAL_CANDIDATE", top_k,
    )
    selected_b1, eligible_b1 = select_weight(b1_grid, constrain_against_zero=True)
    selected_final, eligible_final = select_weight(final_grid, constrain_against_zero=True)

    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(args.results / "reranker_grid.csv", b1_grid + final_grid)
    write_csv(args.results / "reranker_per_query.csv", b1_queries + final_queries)
    audit = {
        "selection_partition": "dev",
        "selection_seeds": seeds,
        "queries": len(features),
        "test_used_for_selection": False,
        "b1_selected": selected_b1,
        "final_selected": selected_final,
        "b1_eligible_lambdas": [row["lambda"] for row in eligible_b1],
        "final_eligible_lambdas": [row["lambda"] for row in eligible_final],
        "reranker_manifest": reranker_manifest,
    }
    write_json(args.results / "selection_reranker.json", audit)

    final = {
        "locked_before_stage7": True,
        "selection_partition": "dev",
        "selection_seeds": seeds,
        "m": int(pre["m"]),
        "r": int(pre["r"]),
        "question_similarity": pre["question_similarity"],
        "retriever": pre["retriever"],
        "k": float(pre["k"]),
        "tau": float(pre["tau"]),
        "reranker": {
            "enabled": selected_final["lambda"] > 0.0,
            "model_id": reranker_manifest["model_id"],
            "revision": reranker_manifest["revision"],
            "fallback_used": bool(reranker_manifest["fallback_used"]),
            "top_k": top_k,
            "lambda": float(selected_final["lambda"]),
        },
        "b1_reranker": {
            "top_k": top_k,
            "lambda": float(selected_b1["lambda"]),
        },
        "dev_metrics": {
            key: float(selected_final[key]) for key in (*SCENARIOS, "mean")
        },
        "selection_constraint": "no_scenario_drop_gt_0.01_vs_previous_step",
        "test_used_for_selection": False,
    }
    Path("configs/final.yaml").write_text(
        yaml.safe_dump(final, sort_keys=False), encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
