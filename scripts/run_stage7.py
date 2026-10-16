"""Run the locked final experiment for ten document splits.

For every seed, sigmoid and reranker parameters are selected on dev before the
test features are built. Test rankings are written as compressed JSONL so the
complete top-50 output remains practical to store and consume in Stage 8.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage6_selection import Resources, bm25_pairs, build_features
from src.data import write_json
from src.methods import (
    descending_indices,
    expanded_weights,
    minmax,
    rerank_top_k,
    rm3_query_weights,
    rrf_scores,
    sigmoid_weight,
    softmax,
)
from src.reranker import load_score_cache, pair_id
from src.semantic import cosine_scores, rrf_pair
from src.text import tokenize


SCENARIOS = ("TPL", "NEW", "PARA", "PNEW")
KS = (0.5, 1.0, 2.0, 4.0, 8.0)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = (
    "B0",
    "B1",
    "B1-DENSE",
    "B1-HYB",
    "B1+RR",
    "RM3",
    "P-EXT",
    "P-QA",
    "P-FUS",
    "ZONE3",
    "SOFT",
    "SOFT-SM",
    "SOFT-QS*",
    "SOFT-QS*-R*",
    "FINAL",
    "ORACLE-W",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_from_order(order: np.ndarray, gold_index: int) -> int:
    position = np.flatnonzero(order == gold_index)
    if len(position) != 1:
        raise ValueError("Gold harus muncul tepat sekali dalam ranking")
    return int(position[0]) + 1


def scenario_mrr(rows: Iterable[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(float(row["rr"]))
    missing = set(SCENARIOS) - set(grouped)
    if missing:
        raise ValueError(f"Skenario tidak lengkap: {sorted(missing)}")
    result = {scenario: float(np.mean(grouped[scenario])) for scenario in SCENARIOS}
    result["mean"] = float(np.mean(list(result.values())))
    return result


def tune_sigmoid(
    features: list[dict], pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, list[dict]]:
    """Tune k and tau on one dev split only."""
    normalized = [(minmax(qa), minmax(ext)) for qa, ext in pairs]
    s_values = np.asarray([feature["s"] for feature in features], dtype=float)
    taus = sorted(set(float(value) for value in np.percentile(s_values, np.arange(5, 100, 5))))
    grid = []
    for k in KS:
        for tau in taus:
            rows = []
            for feature, (qa_score, ext_score) in zip(features, normalized):
                weight = sigmoid_weight(feature["s"], k, tau)
                order = descending_indices(weight * qa_score + (1.0 - weight) * ext_score)
                rank = rank_from_order(order, feature["gold_index"])
                rows.append({"scenario": feature["scenario"], "rr": 1.0 / rank})
            grid.append({"k": k, "tau": tau, **scenario_mrr(rows)})
    selected = max(grid, key=lambda row: (row["mean"], -row["k"], -row["tau"]))
    return selected, grid


def select_reranker_weight(grid: list[dict]) -> tuple[dict, list[dict]]:
    """Apply the no-scenario-drop rule against the pipeline at lambda zero."""
    baseline = next(row for row in grid if float(row["lambda"]) == 0.0)
    eligible = [
        row
        for row in grid
        if all(float(row[scenario]) >= float(baseline[scenario]) - 0.01 for scenario in SCENARIOS)
    ]
    selected = max(eligible, key=lambda row: (row["mean"], -row["lambda"]))
    return selected, eligible


def reranker_inputs(
    features: list[dict], pipeline_scores: list[np.ndarray], score_cache: dict[str, float],
    resources: Resources, top_k: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    prepared = []
    missing = []
    for feature, scores in zip(features, pipeline_scores):
        candidates = descending_indices(scores)[:top_k]
        question = resources.qa_by_qid[feature["qid"]]["question"]
        values = []
        for index in candidates:
            identifier = pair_id(question, feature["patient"].records[int(index)]["text"])
            if identifier not in score_cache:
                missing.append(identifier)
            else:
                values.append(score_cache[identifier])
        if missing:
            continue
        prepared.append((candidates, np.asarray(values, dtype=float)))
    if missing:
        raise KeyError(f"Cache reranker kehilangan {len(set(missing))} pasangan")
    return prepared


def tune_reranker(
    features: list[dict], pipeline_scores: list[np.ndarray], score_cache: dict[str, float],
    resources: Resources, top_k: int,
) -> tuple[dict, list[dict], list[dict]]:
    prepared = reranker_inputs(features, pipeline_scores, score_cache, resources, top_k)
    grid = []
    for weight in LAMBDAS:
        rows = []
        for feature, scores, (candidates, reranker_scores) in zip(
            features, pipeline_scores, prepared,
        ):
            order = rerank_top_k(scores, candidates, reranker_scores, weight)
            rank = rank_from_order(order, feature["gold_index"])
            rows.append({"scenario": feature["scenario"], "rr": 1.0 / rank})
        grid.append({"lambda": weight, **scenario_mrr(rows)})
    selected, eligible = select_reranker_weight(grid)
    return selected, eligible, grid


def softmax_bm25_pairs(
    features: list[dict], m: int, repeat: int, temperature: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for feature in features:
        answer_weights = softmax(feature["q_feedback_scores"][:m], temperature=temperature)
        qa_weights = expanded_weights(
            feature["query_tokens"], feature["q_feedback_tokens"][:m], repeat, answer_weights,
        )
        ext_weights = expanded_weights(
            feature["query_tokens"], feature["ext_feedback_tokens"][:m], repeat,
        )
        pairs.append((
            feature["patient"].model.weighted_scores(qa_weights),
            feature["patient"].model.weighted_scores(ext_weights),
        ))
    return pairs


def rm3_scores(feature: dict, config: dict) -> np.ndarray:
    b1 = feature["patient"].model.scores(feature["query_tokens"])
    base_order = descending_indices(b1)[: int(config["fb_docs"])]
    documents = [
        tokenize(feature["patient"].records[int(index)]["text"], remove_stopwords=True)
        for index in base_order
    ]
    weights = rm3_query_weights(
        feature["query_tokens"], documents, b1[base_order],
        int(config["fb_terms"]), float(config["lambda"]),
    )
    return feature["patient"].model.weighted_scores(weights)


def write_csv_gzip(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test(
    *, seed: int, features: list[dict], pairs: list[tuple[np.ndarray, np.ndarray]],
    resources: Resources, score_cache: dict[str, float], final_config: dict,
    stage5_config: dict, selected_sigmoid: dict, selected_b1_rr: dict,
    selected_final_rr: dict, zone_thresholds: tuple[float, float], output_dir: Path,
) -> dict:
    m, repeat = int(final_config["m"]), int(final_config["r"])
    k, tau = float(selected_sigmoid["k"]), float(selected_sigmoid["tau"])
    top_k = int(final_config["reranker"]["top_k"])
    low_tau, high_tau = zone_thresholds
    sm_pairs = softmax_bm25_pairs(
        features, m, repeat, float(stage5_config["softmax_temperature"]),
    )
    b1_scores = [feature["patient"].model.scores(feature["query_tokens"]) for feature in features]
    final_scores = []
    for feature, (qa_raw, ext_raw) in zip(features, pairs):
        weight = sigmoid_weight(feature["s"], k, tau)
        final_scores.append(weight * minmax(qa_raw) + (1.0 - weight) * minmax(ext_raw))
    b1_rr_inputs = reranker_inputs(features, b1_scores, score_cache, resources, top_k)
    final_rr_inputs = reranker_inputs(features, final_scores, score_cache, resources, top_k)

    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"seed_{seed:02d}_rankings.jsonl.gz"
    query_path = output_dir / f"seed_{seed:02d}_queries.csv.gz"
    query_rows = []
    ranking_rows = 0
    short_documents = set()
    with gzip.open(ranking_path, "wt", encoding="utf-8", newline="\n") as handle:
        for number, (feature, pair, sm_pair) in enumerate(zip(features, pairs, sm_pairs)):
            qa_raw, ext_raw = pair
            qa_norm, ext_norm = minmax(qa_raw), minmax(ext_raw)
            b0 = feature["patient"].model.scores(tokenize(feature["query"]))
            b1 = b1_scores[number]
            dense = cosine_scores(feature["query_vector"], feature["patient_vectors"])
            b1_hybrid = rrf_pair(b1, dense, k=60)
            qa_order, ext_order = descending_indices(qa_raw), descending_indices(ext_raw)
            fused = rrf_scores([qa_order, ext_order], len(qa_order), k=60)
            weight = sigmoid_weight(feature["s"], k, tau)
            soft_score = final_scores[number]
            if feature["s"] >= high_tau:
                zone_score = qa_raw
            elif feature["s"] >= low_tau:
                zone_score = fused
            else:
                zone_score = ext_raw
            sm_qa, sm_ext = sm_pair
            softmax_score = weight * minmax(sm_qa) + (1.0 - weight) * minmax(sm_ext)
            rm3 = rm3_scores(feature, stage5_config["rm3"])

            b1_candidates, b1_reranker = b1_rr_inputs[number]
            final_candidates, final_reranker = final_rr_inputs[number]
            b1_rr_order = rerank_top_k(
                b1, b1_candidates, b1_reranker, float(selected_b1_rr["lambda"]),
            )
            final_order = rerank_top_k(
                soft_score, final_candidates, final_reranker,
                float(selected_final_rr["lambda"]),
            )
            orders = {
                "B0": descending_indices(b0),
                "B1": descending_indices(b1),
                "B1-DENSE": descending_indices(dense),
                "B1-HYB": descending_indices(b1_hybrid),
                "B1+RR": b1_rr_order,
                "RM3": descending_indices(rm3),
                "P-EXT": ext_order,
                "P-QA": qa_order,
                "P-FUS": descending_indices(fused),
                "ZONE3": descending_indices(zone_score),
                "SOFT": descending_indices(soft_score),
                "SOFT-SM": descending_indices(softmax_score),
                "SOFT-QS*": descending_indices(soft_score),
                "SOFT-QS*-R*": descending_indices(soft_score),
                "FINAL": final_order,
            }
            qa_rank = rank_from_order(qa_order, feature["gold_index"])
            ext_rank = rank_from_order(ext_order, feature["gold_index"])
            orders["ORACLE-W"] = qa_order if qa_rank <= ext_rank else ext_order
            if tuple(orders) != METHODS:
                raise AssertionError("Daftar metode run final tidak lengkap atau tidak berurutan")

            ranks = {
                method: rank_from_order(order, feature["gold_index"])
                for method, order in orders.items()
            }
            query_rows.append({
                "seed": seed,
                "partition": "test",
                "qid": feature["qid"],
                "doc_id": feature["doc_id"],
                "scenario": feature["scenario"],
                "question_type": feature["question_type"],
                "s": feature["s"],
                "w": weight,
                "feedback_type_matches": sum(
                    resources.qa_by_qid[qid]["question_type"] == feature["question_type"]
                    for qid in feature["q_feedback_qids"][:m]
                ),
                **{f"rank__{method}": rank for method, rank in ranks.items()},
            })
            item_ids = feature["patient"].item_ids
            if len(item_ids) < 50:
                short_documents.add(feature["doc_id"])
            for method, order in orders.items():
                ranking = [item_ids[int(index)] for index in order[:50]]
                if len(ranking) != min(50, len(item_ids)) or len(ranking) != len(set(ranking)):
                    raise AssertionError("Ranking top-50 tidak lengkap atau mengandung duplikat")
                handle.write(json.dumps({
                    "seed": seed,
                    "partition": "test",
                    "qid": feature["qid"],
                    "doc_id": feature["doc_id"],
                    "scenario": feature["scenario"],
                    "question_type": feature["question_type"],
                    "method": method,
                    "gold_chunk_id": feature["gold_chunk_id"],
                    "available_chunks": len(item_ids),
                    "ranking": ranking,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                ranking_rows += 1
    write_csv_gzip(query_path, query_rows)
    return {
        "test_queries": len(features),
        "methods": len(METHODS),
        "ranking_rows": ranking_rows,
        "short_documents": len(short_documents),
        "ranking_file": str(ranking_path),
        "ranking_sha256": file_sha256(ranking_path),
        "query_file": str(query_path),
        "query_sha256": file_sha256(query_path),
    }


def write_grid(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(
    seed: int, *, resources: Resources, processed: Path, score_cache: dict[str, float],
    final_config: dict, stage5_config: dict, results: Path,
) -> dict:
    started = time.perf_counter()
    qs_method = final_config["question_similarity"]
    if qs_method != "QS-BM25" or final_config["retriever"] != "R-BM25":
        raise NotImplementedError("Run Tahap 7 ini dikunci untuk komponen FINAL yang terpilih")
    m, repeat = int(final_config["m"]), int(final_config["r"])

    # All selection happens before test features are constructed.
    dev = build_features(resources, processed, [seed], qs_method, partition="dev")
    dev_pairs = bm25_pairs(dev, m, repeat)
    selected_sigmoid, sigmoid_grid = tune_sigmoid(dev, dev_pairs)
    dev_b1 = [feature["patient"].model.scores(feature["query_tokens"]) for feature in dev]
    dev_final = []
    for feature, (qa_raw, ext_raw) in zip(dev, dev_pairs):
        weight = sigmoid_weight(feature["s"], selected_sigmoid["k"], selected_sigmoid["tau"])
        dev_final.append(weight * minmax(qa_raw) + (1.0 - weight) * minmax(ext_raw))
    selected_b1_rr, eligible_b1, b1_grid = tune_reranker(
        dev, dev_b1, score_cache, resources, int(final_config["b1_reranker"]["top_k"]),
    )
    selected_final_rr, eligible_final, final_grid = tune_reranker(
        dev, dev_final, score_cache, resources, int(final_config["reranker"]["top_k"]),
    )
    low_tau, high_tau = (
        float(value) for value in np.percentile(
            [feature["s"] for feature in dev],
            [stage5_config["zone3"]["tau2_percentile"], stage5_config["zone3"]["tau1_percentile"]],
        )
    )

    grid_dir = results / "params" / "grids"
    write_grid(grid_dir / f"seed_{seed:02d}_sigmoid.csv", sigmoid_grid)
    write_grid(
        grid_dir / f"seed_{seed:02d}_reranker.csv",
        [{"pipeline": "B1", **row} for row in b1_grid]
        + [{"pipeline": "FINAL", **row} for row in final_grid],
    )

    # Test is built only after every tunable value for this seed has been selected.
    test = build_features(resources, processed, [seed], qs_method, partition="test")
    test_pairs = bm25_pairs(test, m, repeat)
    output = evaluate_test(
        seed=seed,
        features=test,
        pairs=test_pairs,
        resources=resources,
        score_cache=score_cache,
        final_config=final_config,
        stage5_config=stage5_config,
        selected_sigmoid=selected_sigmoid,
        selected_b1_rr=selected_b1_rr,
        selected_final_rr=selected_final_rr,
        zone_thresholds=(low_tau, high_tau),
        output_dir=results / "runs",
    )
    parameters = {
        "seed": seed,
        "selection_partition": "dev",
        "evaluation_partition": "test",
        "test_used_for_tuning": False,
        "dev_queries": len(dev),
        "test_queries": len(test),
        "m": m,
        "r": repeat,
        "question_similarity": qs_method,
        "retriever": final_config["retriever"],
        "sigmoid": {
            "k": float(selected_sigmoid["k"]),
            "tau": float(selected_sigmoid["tau"]),
            "dev_metrics": {
                key: float(selected_sigmoid[key]) for key in (*SCENARIOS, "mean")
            },
        },
        "zone3": {"tau2": low_tau, "tau1": high_tau},
        "rm3": stage5_config["rm3"],
        "b1_reranker": {
            "lambda": float(selected_b1_rr["lambda"]),
            "eligible_lambdas": [float(row["lambda"]) for row in eligible_b1],
            "dev_metrics": {
                key: float(selected_b1_rr[key]) for key in (*SCENARIOS, "mean")
            },
        },
        "final_reranker": {
            "lambda": float(selected_final_rr["lambda"]),
            "eligible_lambdas": [float(row["lambda"]) for row in eligible_final],
            "dev_metrics": {
                key: float(selected_final_rr[key]) for key in (*SCENARIOS, "mean")
            },
        },
        "outputs": output,
        "elapsed_seconds": time.perf_counter() - started,
    }
    param_path = results / "params" / f"seed_{seed:02d}.yaml"
    param_path.parent.mkdir(parents=True, exist_ok=True)
    param_path.write_text(yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8")
    print(json.dumps({
        "seed": seed,
        "k": parameters["sigmoid"]["k"],
        "tau": parameters["sigmoid"]["tau"],
        "b1_lambda": parameters["b1_reranker"]["lambda"],
        "final_lambda": parameters["final_reranker"]["lambda"],
        "test_queries": output["test_queries"],
        "ranking_rows": output["ranking_rows"],
        "elapsed_seconds": parameters["elapsed_seconds"],
    }), flush=True)
    return parameters


def load_completed_seed(results: Path, seed: int) -> dict:
    """Load and verify a completed seed before resuming a long final run."""
    param_path = results / "params" / f"seed_{seed:02d}.yaml"
    parameters = yaml.safe_load(param_path.read_text(encoding="utf-8"))
    if parameters["seed"] != seed or parameters.get("test_used_for_tuning") is not False:
        raise ValueError(f"Parameter seed {seed} tidak valid untuk resume")
    outputs = parameters["outputs"]
    for file_key, hash_key in (
        ("ranking_file", "ranking_sha256"),
        ("query_file", "query_sha256"),
    ):
        path = Path(outputs[file_key])
        if not path.exists() or file_sha256(path) != outputs[hash_key]:
            raise ValueError(f"Artefak resume seed {seed} gagal verifikasi: {path}")
    return parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--score-cache", type=Path, default=Path("models/reranker/stage6_scores.json"))
    parser.add_argument("--reranker-manifest", type=Path, default=Path("models/reranker/stage6_manifest.json"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    final_path = Path("configs/final.yaml")
    final_config = yaml.safe_load(final_path.read_text(encoding="utf-8"))
    stage5_config = yaml.safe_load(Path("configs/stage5.yaml").read_text(encoding="utf-8"))
    if not final_config.get("locked_before_stage7") or final_config.get("test_used_for_selection"):
        raise ValueError("Konfigurasi FINAL belum terkunci secara sah sebelum Tahap 7")
    manifest = json.loads(args.reranker_manifest.read_text(encoding="utf-8"))
    if (
        manifest["model_id"] != final_config["reranker"]["model_id"]
        or manifest["revision"] != final_config["reranker"]["revision"]
    ):
        raise ValueError("Manifest cache reranker tidak cocok dengan konfigurasi FINAL")
    score_cache = load_score_cache(args.score_cache)
    if len(score_cache) != int(manifest["pairs"]) or not np.isfinite(list(score_cache.values())).all():
        raise ValueError("Cache reranker tidak lengkap atau memuat skor non-finite")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in range(1, 11) for seed in args.seeds):
        raise ValueError("Seed harus unik dan berada dalam rentang 1--10")

    resources = Resources(args.processed, args.embeddings)
    started = time.perf_counter()
    runs = []
    resumed_seeds = []
    for seed in args.seeds:
        param_path = args.results / "params" / f"seed_{seed:02d}.yaml"
        if args.resume and param_path.exists():
            print(f"Resuming verified seed {seed}", flush=True)
            runs.append(load_completed_seed(args.results, seed))
            resumed_seeds.append(seed)
            continue
        print(f"Running seed {seed}", flush=True)
        runs.append(run_seed(
            seed,
            resources=resources,
            processed=args.processed,
            score_cache=score_cache,
            final_config=final_config,
            stage5_config=stage5_config,
            results=args.results,
        ))
    audit = {
        "status": "PASS",
        "seeds": args.seeds,
        "completed_seeds": len(runs),
        "resumed_seeds": resumed_seeds,
        "methods": list(METHODS),
        "selection_partition": "dev",
        "evaluation_partition": "test",
        "test_used_for_tuning": False,
        "final_config_sha256": file_sha256(final_path),
        "reranker_model_id": manifest["model_id"],
        "reranker_revision": manifest["revision"],
        "reranker_fallback_used": bool(manifest["fallback_used"]),
        "reranker_pairs": len(score_cache),
        "total_test_queries": sum(run["test_queries"] for run in runs),
        "total_ranking_rows": sum(run["outputs"]["ranking_rows"] for run in runs),
        "elapsed_seconds": time.perf_counter() - started,
        "runs": [
            {
                "seed": run["seed"],
                "k": run["sigmoid"]["k"],
                "tau": run["sigmoid"]["tau"],
                "b1_lambda": run["b1_reranker"]["lambda"],
                "final_lambda": run["final_reranker"]["lambda"],
                "test_queries": run["test_queries"],
                **run["outputs"],
            }
            for run in runs
        ],
    }
    write_json(args.results / "stage7" / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
