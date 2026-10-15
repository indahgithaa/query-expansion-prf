"""Run Stage 5 BM25 baselines, classical PRF, core methods, and dev tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bm25 import BM25, seeded_tie_rank
from src.data import QUESTION_TYPE_ORDER, read_json, read_jsonl, write_json
from src.methods import (
    descending_indices, expanded_weights, minmax, rank_of, rm3_query_weights,
    rrf_scores, sigmoid_weight, softmax,
)
from src.sources import ExternalFeedbackIndex, PatientBM25Index, QuestionFeedbackIndex
from src.text import tokenize


def mean_by_scenario(rows: list[dict], key: str = "rr") -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(float(row[key]))
    result = {scenario: float(np.mean(grouped[scenario])) for scenario in ("TPL", "NEW", "PARA", "PNEW")}
    result["mean"] = float(np.mean(list(result.values())))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_features(processed: Path, seeds: list[int]) -> tuple[list[dict], dict]:
    qa = read_jsonl(processed / "qa.jsonl")
    chunks = read_jsonl(processed / "chunks.jsonl")
    by_doc = defaultdict(list)
    for row in chunks:
        by_doc[row["doc_id"]].append(row)
    patient = {doc_id: PatientBM25Index(rows) for doc_id, rows in by_doc.items()}
    all_features, audit = [], {"seeds": []}

    for seed in seeds:
        split = read_json(processed / "splits" / f"seed_{seed:02d}.json")
        collection_qids = set(split["qids"]["collection"])
        collection_docs = set(split["documents"]["collection"])
        collection_qa = [row for row in qa if row["qid"] in collection_qids]
        collection_chunks = [row for row in chunks if row["doc_id"] in collection_docs]
        q_indexes = {None: QuestionFeedbackIndex(collection_qa, tie_seed=seed * 1009)}
        for question_type in QUESTION_TYPE_ORDER:
            q_indexes[question_type] = QuestionFeedbackIndex(
                [row for row in collection_qa if row["question_type"] != question_type],
                tie_seed=seed * 1009,
            )
        external = ExternalFeedbackIndex(collection_chunks)
        external_cache = {}
        scenario_rows = [
            row for row in read_jsonl(processed / "scenarios" / f"seed_{seed:02d}.jsonl")
            if row["partition"] == "dev"
        ]
        for number, row in enumerate(scenario_rows, 1):
            scenario = row["scenario"]
            exclusion = row["question_type"] if scenario in {"NEW", "PNEW"} else None
            q_index = q_indexes[exclusion]
            q_tokens = tokenize(row["query"], remove_stopwords=True)
            q_scores = q_index.model.scores(tokenize(row["query"]))
            q_ranking = seeded_tie_rank(
                q_scores, q_index.item_ids, seed=q_index.tie_seed,
                query_id=f"{row['qid']}:{scenario}", top_k=5,
            )
            q_feedback = [q_index.records[item.index] for item in q_ranking]
            q_feedback_scores = [float(item.score) for item in q_ranking]
            ext_key = row["query"]
            if ext_key not in external_cache:
                scores = external.model.scores(q_tokens)
                order = descending_indices(scores)[:5]
                external_cache[ext_key] = [external.records[int(index)] for index in order]
            ext_feedback = external_cache[ext_key]
            pindex = patient[row["doc_id"]]
            gold_index = pindex.item_ids.index(row["gold_chunk_id"])
            b0 = pindex.model.scores(tokenize(row["query"]))
            b1 = pindex.model.scores(q_tokens)
            all_features.append({
                **row, "patient": pindex, "gold_index": gold_index, "query_tokens": q_tokens,
                "q_feedback_tokens": [tokenize(" ".join(item["answer_texts"]), remove_stopwords=True) for item in q_feedback],
                "q_feedback_scores": q_feedback_scores,
                "q_feedback_types": [item["question_type"] for item in q_feedback],
                "q_feedback_qids": [item["qid"] for item in q_feedback],
                "ext_feedback_tokens": [tokenize(item["text"], remove_stopwords=True) for item in ext_feedback],
                "ext_feedback_ids": [item["chunk_id"] for item in ext_feedback],
                "s": float(q_feedback_scores[0]), "b0": b0, "b1": b1,
            })
        audit["seeds"].append({"seed": seed, "dev_rows": len(scenario_rows), "external_query_cache": len(external_cache)})
    return all_features, audit


def expanded_pair(feature: dict, m: int, repeat: int, answer_softmax: bool = False) -> tuple[np.ndarray, np.ndarray]:
    answer_weights = None
    if answer_softmax:
        answer_weights = softmax(feature["q_feedback_scores"][:m], temperature=1.0)
    qa_weights = expanded_weights(
        feature["query_tokens"], feature["q_feedback_tokens"][:m], repeat, answer_weights,
    )
    ext_weights = expanded_weights(feature["query_tokens"], feature["ext_feedback_tokens"][:m], repeat)
    return (
        feature["patient"].model.weighted_scores(qa_weights),
        feature["patient"].model.weighted_scores(ext_weights),
    )


def tune_soft(features: list[dict], m: int, repeat: int) -> tuple[dict, list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    pairs = [tuple(minmax(x) for x in expanded_pair(feature, m, repeat)) for feature in features]
    s_values = np.asarray([feature["s"] for feature in features])
    taus = sorted(set(float(x) for x in np.percentile(s_values, np.arange(5, 100, 5))))
    rows = []
    for k in (0.5, 1.0, 2.0, 4.0, 8.0):
        for tau in taus:
            per_query = []
            for feature, (qa_score, ext_score) in zip(features, pairs):
                weight = sigmoid_weight(feature["s"], k, tau)
                rank = rank_of(weight * qa_score + (1.0 - weight) * ext_score, feature["gold_index"])
                per_query.append({"scenario": feature["scenario"], "rr": 1.0 / rank})
            metrics = mean_by_scenario(per_query)
            rows.append({"m": m, "r": repeat, "k": k, "tau": tau, **metrics})
    best = max(rows, key=lambda row: (row["mean"], -row["k"], -row["tau"]))
    return best, rows, pairs


def evaluate_rm3(features: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    for fb_docs in (3, 5, 10):
        for fb_terms in (10, 20, 30):
            for interpolation in (0.3, 0.5, 0.7):
                per_query = []
                for feature in features:
                    base_order = descending_indices(feature["b1"])[:fb_docs]
                    documents = [tokenize(feature["patient"].records[int(i)]["text"], remove_stopwords=True) for i in base_order]
                    weights = rm3_query_weights(
                        feature["query_tokens"], documents, feature["b1"][base_order], fb_terms, interpolation,
                    )
                    score = feature["patient"].model.weighted_scores(weights)
                    rank = rank_of(score, feature["gold_index"])
                    per_query.append({"scenario": feature["scenario"], "rr": 1.0 / rank})
                metrics = mean_by_scenario(per_query)
                rows.append({"fb_docs": fb_docs, "fb_terms": fb_terms, "lambda": interpolation, **metrics})
    return max(rows, key=lambda row: row["mean"]), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results/stage5"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    features, audit = build_features(args.processed, args.seeds)

    candidates, tuning_rows, pair_cache = [], [], {}
    for m in (1, 3, 5):
        for repeat in (1, 3, 5):
            best, grid, pairs = tune_soft(features, m, repeat)
            candidates.append(best)
            tuning_rows.extend(grid)
            pair_cache[(m, repeat)] = pairs
    default = next(row for row in candidates if row["m"] == 3 and row["r"] == 3)
    eligible = [
        row for row in candidates
        if all(row[s] >= default[s] - 0.01 for s in ("TPL", "NEW", "PARA", "PNEW"))
    ]
    selected = max(eligible, key=lambda row: row["mean"])
    m, repeat, k, tau = int(selected["m"]), int(selected["r"]), float(selected["k"]), float(selected["tau"])
    pairs = pair_cache[(m, repeat)]
    low_tau, high_tau = np.percentile([feature["s"] for feature in features], [33, 67])

    method_rows = defaultdict(list)
    query_rows = []
    for feature, (qa_score, ext_score) in zip(features, pairs):
        gold = feature["gold_index"]
        qa_raw, ext_raw = expanded_pair(feature, m, repeat)
        qa_order, ext_order = descending_indices(qa_raw), descending_indices(ext_raw)
        fused = rrf_scores([qa_order, ext_order], len(qa_order), k=60)
        weight = sigmoid_weight(feature["s"], k, tau)
        soft_score = weight * qa_score + (1.0 - weight) * ext_score
        if feature["s"] >= high_tau:
            zone_score = qa_raw
        elif feature["s"] >= low_tau:
            zone_score = fused
        else:
            zone_score = ext_raw
        sm_qa_raw, sm_ext_raw = expanded_pair(feature, m, repeat, answer_softmax=True)
        sm_score = weight * minmax(sm_qa_raw) + (1.0 - weight) * minmax(sm_ext_raw)
        scores = {
            "B0": feature["b0"], "B1": feature["b1"], "P-EXT": ext_raw, "P-QA": qa_raw,
            "P-FUS": fused, "ZONE3": zone_score, "SOFT": soft_score, "SOFT-SM": sm_score,
        }
        ranks = {method: rank_of(score, gold) for method, score in scores.items()}
        ranks["ORACLE-W"] = min(ranks["P-QA"], ranks["P-EXT"])
        for method, rank in ranks.items():
            method_rows[method].append({"scenario": feature["scenario"], "rr": 1.0 / rank})
        query_rows.append({
            "seed": feature["seed"], "qid": feature["qid"], "scenario": feature["scenario"],
            "question_type": feature["question_type"], "s": feature["s"], "w": weight,
            "feedback_type_matches": sum(t == feature["question_type"] for t in feature["q_feedback_types"][:m]),
            **{f"rank_{method}": rank for method, rank in ranks.items()},
        })

    rm3_best, rm3_grid = evaluate_rm3(features)
    for feature in features:
        base_order = descending_indices(feature["b1"])[: int(rm3_best["fb_docs"])]
        documents = [tokenize(feature["patient"].records[int(i)]["text"], remove_stopwords=True) for i in base_order]
        weights = rm3_query_weights(
            feature["query_tokens"], documents, feature["b1"][base_order],
            int(rm3_best["fb_terms"]), float(rm3_best["lambda"]),
        )
        rank = rank_of(feature["patient"].model.weighted_scores(weights), feature["gold_index"])
        method_rows["RM3"].append({"scenario": feature["scenario"], "rr": 1.0 / rank})

    metrics_rows = [{"method": method, **mean_by_scenario(rows)} for method, rows in method_rows.items()]
    write_csv(args.results / "base_tuning.csv", candidates)
    write_csv(args.results / "soft_grid.csv", tuning_rows)
    write_csv(args.results / "rm3_grid.csv", rm3_grid)
    write_csv(args.results / "dev_metrics.csv", sorted(metrics_rows, key=lambda row: row["method"]))
    write_csv(args.results / "per_query.csv", query_rows)

    heat = [row for row in tuning_rows if row["m"] == m and row["r"] == repeat]
    ks = sorted(set(row["k"] for row in heat)); taus = sorted(set(row["tau"] for row in heat))
    matrix = np.asarray([[next(row["mean"] for row in heat if row["k"] == kval and row["tau"] == t) for t in taus] for kval in ks])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(ks)), labels=ks); ax.set_ylabel("k")
    ax.set_xticks(range(len(taus)), labels=[f"{x:.2f}" for x in taus], rotation=60, ha="right")
    ax.set_xlabel("tau (persentil skor s)"); ax.set_title("Sensitivitas MRR dev SOFT")
    fig.colorbar(image, ax=ax, label="MRR rata-rata")
    fig.tight_layout(); fig.savefig("reports/figures/05_soft_heatmap.png", dpi=300); plt.close(fig)

    config = {
        "selection_seeds": args.seeds,
        "base": {"m": m, "r": repeat, "k": k, "tau": tau},
        "zone3": {"tau2_percentile": 33, "tau1_percentile": 67, "tau2": float(low_tau), "tau1": float(high_tau)},
        "softmax_temperature": 1.0,
        "rm3": {key: rm3_best[key] for key in ("fb_docs", "fb_terms", "lambda")},
        "selection_constraint": "no_scenario_drop_gt_0.01_vs_m3_r3",
    }
    Path("configs/stage5.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(args.results / "audit.json", {"features": len(features), "selected": selected, "rm3": rm3_best, **audit})
    print(json.dumps({"features": len(features), "selected": selected, "rm3": rm3_best, "metrics": metrics_rows}, indent=2))


if __name__ == "__main__":
    main()
