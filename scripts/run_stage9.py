"""Analyze question similarity, retrieval, reranking, and component contributions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage6_selection import (
    SCENARIOS,
    Resources,
    bm25_pairs,
    build_features,
    tune_soft,
    write_csv,
)
from scripts.run_stage7 import file_sha256
from src.data import write_json


QS_METHODS = (
    "QS-BM25",
    "QS-EMB",
    "QS-HYB-MM",
    "QS-HYB-RRF",
    "QS-TFIDF-WORD",
    "QS-TFIDF-CHAR",
)
DECOMPOSITION = (
    ("Baseline query", "B1"),
    ("+ QA feedback expansion", "P-QA"),
    ("+ external feedback and fusion", "P-FUS"),
    ("+ adaptive sigmoid weighting", "SOFT"),
    ("+ selected question similarity", "SOFT-QS*"),
    ("+ selected chunk retriever", "SOFT-QS*-R*"),
    ("+ cross-encoder reranker", "FINAL"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_question_similarity(
    resources: Resources, processed: Path, results: Path, output: Path, m: int, repeat: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    summaries: list[dict] = []
    feedback_rows: list[dict] = []
    score_rows: list[dict] = []
    original = {
        row["qs_method"]: row
        for row in read_csv(results / "stage6" / "qs_candidates.csv")
    }

    for method in QS_METHODS:
        print(f"Analyzing {method}", flush=True)
        features = build_features(resources, processed, [1, 2, 3], method, partition="dev")
        pairs = bm25_pairs(features, m, repeat)
        best, _ = tune_soft(features, pairs, {"qs_method": method, "retriever": "R-BM25"})
        summaries.append(best)
        if method in original:
            for key in (*SCENARIOS, "mean"):
                if not np.isclose(float(best[key]), float(original[method][key]), atol=1e-12):
                    raise AssertionError(f"Reproduksi Stage 6 berubah: {method}, {key}")

        for feature in features:
            score_rows.append(
                {
                    "method": method,
                    "seed": feature["seed"],
                    "qid": feature["qid"],
                    "scenario": feature["scenario"],
                    "s": feature["s"],
                }
            )
        for scenario in ("TPL", "PARA"):
            selected = [feature for feature in features if feature["scenario"] == scenario]
            matches = sum(
                resources.qa_by_qid[qid]["question_type"] == feature["question_type"]
                for feature in selected
                for qid in feature["q_feedback_qids"][:m]
            )
            total = sum(min(m, len(feature["q_feedback_qids"])) for feature in selected)
            feedback_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "queries": len(selected),
                    "matching_feedback": matches,
                    "feedback_items": total,
                    "accuracy": matches / total,
                }
            )

    write_csv(output / "question_similarity.csv", summaries)
    write_csv(output / "feedback_accuracy.csv", feedback_rows)
    write_csv(output / "similarity_scores.csv", score_rows)
    return summaries, feedback_rows, score_rows


def plot_score_distributions(score_rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    for axis, method in zip(axes.flat, QS_METHODS, strict=True):
        values = [
            [float(row["s"]) for row in score_rows if row["method"] == method and row["scenario"] == scenario]
            for scenario in SCENARIOS
        ]
        box = axis.boxplot(values, tick_labels=SCENARIOS, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axis.set_title(method)
        axis.set_ylabel("Top-1 similarity score s")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Distribusi skor question similarity pada dev seed 1-3", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_retrievers(results: Path, output: Path) -> list[dict]:
    rows = read_csv(results / "stage6" / "retriever_candidates.csv")
    lookup = {row["retriever"]: row for row in rows}
    dense_rocchio = max(
        (row for row in rows if row["retriever"].startswith("R-DENSE-ROCCHIO")),
        key=lambda row: float(row["mean"]),
    )
    hybrid_rocchio = max(
        (row for row in rows if row["retriever"].startswith("R-HYB-ROCCHIO")),
        key=lambda row: float(row["mean"]),
    )
    selected = [
        {"expansion": "BM25 weighted query", **lookup["R-BM25"]},
        {"expansion": "Full expanded text", **lookup["R-DENSE-FULL"]},
        {"expansion": "Full expanded text", **lookup["R-HYB-FULL"]},
        {"expansion": "Rocchio best", **dense_rocchio},
        {"expansion": "Rocchio best", **hybrid_rocchio},
    ]
    write_csv(output / "retriever_analysis.csv", selected)
    return selected


def analyze_test_components(output: Path, results: Path) -> tuple[list[dict], list[dict], dict]:
    strict = read_csv(results / "stage8" / "strict_summary.csv")
    by_key = {(row["method"], row["scenario"]): row for row in strict}
    reranker = []
    for method in ("B1", "B1+RR", "SOFT", "FINAL"):
        label = {"B1+RR": "B1 + reranker", "FINAL": "SOFT + reranker"}.get(method, method)
        for scenario in (*SCENARIOS, "average"):
            row = by_key[(method, scenario)]
            reranker.append(
                {
                    "pipeline": label,
                    "method": method,
                    "scenario": scenario,
                    "mrr_mean": row["mrr_mean"],
                    "mrr_sd": row["mrr_sd"],
                }
            )
    write_csv(output / "reranker_analysis.csv", reranker)

    decomposition = []
    previous = None
    for stage, method in DECOMPOSITION:
        values = {
            scenario: float(by_key[(method, scenario)]["mrr_mean"])
            for scenario in (*SCENARIOS, "average")
        }
        decomposition.append(
            {
                "stage": stage,
                "method": method,
                **values,
                "delta_average_vs_previous": "" if previous is None else values["average"] - previous,
            }
        )
        previous = values["average"]
    total_gain = decomposition[-1]["average"] - decomposition[0]["average"]
    increments = sum(float(row["delta_average_vs_previous"] or 0.0) for row in decomposition)
    if not np.isclose(total_gain, increments, atol=1e-12):
        raise AssertionError("Dekomposisi tidak menjumlah ke kenaikan B1--FINAL")
    write_csv(output / "decomposition.csv", decomposition)

    b1_lambdas = Counter()
    final_lambdas = Counter()
    for seed in range(1, 11):
        parameters = yaml.safe_load((results / "params" / f"seed_{seed:02d}.yaml").read_text(encoding="utf-8"))
        if parameters.get("test_used_for_tuning") is not False:
            raise AssertionError(f"Audit dev/test gagal pada seed {seed}")
        b1_lambdas[float(parameters["b1_reranker"]["lambda"])] += 1
        final_lambdas[float(parameters["final_reranker"]["lambda"])] += 1
    lambda_summary = {
        "B1+RR": {str(key): value for key, value in sorted(b1_lambdas.items())},
        "FINAL": {str(key): value for key, value in sorted(final_lambdas.items())},
    }
    write_json(output / "lambda_selection.json", lambda_summary)
    return reranker, decomposition, lambda_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("reports/figures/09_similarity_score_distribution.png"),
    )
    parser.add_argument(
        "--reuse-question-analysis", action="store_true",
        help="Reuse completed question-similarity CSVs after an interrupted post-processing step.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.results / "stage9"
    output.mkdir(parents=True, exist_ok=True)

    stage8 = json.loads((args.results / "stage8" / "verification.json").read_text(encoding="utf-8"))
    if stage8.get("status") != "PASS":
        raise AssertionError("Stage 8 verification must pass before Stage 9")
    final_config = yaml.safe_load(Path("configs/final.yaml").read_text(encoding="utf-8"))
    if final_config.get("test_used_for_selection") is not False:
        raise AssertionError("FINAL configuration is not selection-safe")
    resources = Resources(args.processed, args.embeddings)
    cached = (
        output / "question_similarity.csv",
        output / "feedback_accuracy.csv",
        output / "similarity_scores.csv",
    )
    if args.reuse_question_analysis:
        if not all(path.is_file() for path in cached):
            raise FileNotFoundError("Question-similarity cache is incomplete")
        summaries, feedback, scores = (read_csv(path) for path in cached)
        if (
            {row["qs_method"] for row in summaries} != set(QS_METHODS)
            or len(feedback) != len(QS_METHODS) * 2
            or len(scores) != 2_052 * len(QS_METHODS)
        ):
            raise AssertionError("Question-similarity cache failed coverage checks")
    else:
        summaries, feedback, scores = analyze_question_similarity(
            resources, args.processed, args.results, output,
            int(final_config["m"]), int(final_config["r"]),
        )
    plot_score_distributions(scores, args.figure)
    retrievers = analyze_retrievers(args.results, output)
    reranker, decomposition, lambdas = analyze_test_components(output, args.results)

    output_files = [
        output / "question_similarity.csv",
        output / "feedback_accuracy.csv",
        output / "similarity_scores.csv",
        output / "retriever_analysis.csv",
        output / "reranker_analysis.csv",
        output / "decomposition.csv",
        output / "lambda_selection.json",
        args.figure,
    ]
    audit = {
        "status": "PASS",
        "selection_analysis_partition": "dev",
        "selection_analysis_seeds": [1, 2, 3],
        "final_component_analysis_partition": "test",
        "final_component_analysis_seeds": list(range(1, 11)),
        "test_used_for_selection": False,
        "question_similarity_methods": list(QS_METHODS),
        "question_similarity_rows": len(summaries),
        "feedback_accuracy_rows": len(feedback),
        "similarity_score_rows": len(scores),
        "retriever_rows": len(retrievers),
        "reranker_rows": len(reranker),
        "decomposition_rows": len(decomposition),
        "lambda_selection": lambdas,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in output_files
        },
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
