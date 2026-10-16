"""Aggregate Stage 7 runs into Stage 8 metrics and paired statistics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage7 import METHODS, SCENARIOS
from src.data import read_jsonl, write_json
from src.metrics import (
    average_precision,
    binary_ndcg_at_k,
    holm_adjust,
    rank_biserial_effect,
    recall_at_k,
    reciprocal_rank,
    robustness_counts,
    single_gold_ndcg,
)


STRICT_METRICS = ("mrr", "recall_1", "recall_3", "recall_5", "recall_10", "ndcg_10")
COMPARISONS = (
    ("FINAL", "B1"),
    ("FINAL", "B1+RR"),
    ("FINAL", "RM3"),
    ("SOFT", "B1"),
    ("SOFT", "P-QA"),
    ("SOFT", "P-FUS"),
    ("SOFT", "ZONE3"),
)
THRESHOLDS = ("0.5", "0.6", "0.7")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261016


def read_query_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty vector")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def strict_values(ranks: np.ndarray) -> dict[str, float]:
    return {
        "mrr": float(np.mean([reciprocal_rank(int(rank)) for rank in ranks])),
        "recall_1": float(np.mean([recall_at_k(int(rank), 1) for rank in ranks])),
        "recall_3": float(np.mean([recall_at_k(int(rank), 3) for rank in ranks])),
        "recall_5": float(np.mean([recall_at_k(int(rank), 5) for rank in ranks])),
        "recall_10": float(np.mean([recall_at_k(int(rank), 10) for rank in ranks])),
        "ndcg_10": float(np.mean([single_gold_ndcg(int(rank), 10) for rank in ranks])),
    }


def build_strict(query_rows: dict[int, list[dict[str, str]]]) -> tuple[list[dict], list[dict]]:
    per_seed: list[dict] = []
    for seed, rows in query_rows.items():
        for method in METHODS:
            scenario_rows: list[dict] = []
            for scenario in SCENARIOS:
                ranks = np.asarray(
                    [int(row[f"rank__{method}"]) for row in rows if row["scenario"] == scenario],
                    dtype=int,
                )
                metric_values = strict_values(ranks)
                result = {"seed": seed, "method": method, "scenario": scenario, "n": len(ranks), **metric_values}
                per_seed.append(result)
                scenario_rows.append(result)
            per_seed.append(
                {
                    "seed": seed,
                    "method": method,
                    "scenario": "average",
                    "n": sum(row["n"] for row in scenario_rows),
                    **{
                        metric: float(np.mean([row[metric] for row in scenario_rows]))
                        for metric in STRICT_METRICS
                    },
                }
            )

    summary: list[dict] = []
    for method in METHODS:
        for scenario in (*SCENARIOS, "average"):
            selected = [row for row in per_seed if row["method"] == method and row["scenario"] == scenario]
            result: dict[str, float | int | str] = {
                "method": method,
                "scenario": scenario,
                "seeds": len(selected),
                "n_total": sum(int(row["n"]) for row in selected),
            }
            for metric in STRICT_METRICS:
                result[f"{metric}_mean"], result[f"{metric}_sd"] = mean_sd(row[metric] for row in selected)
            summary.append(result)
    return per_seed, summary


def load_multi_qrels(processed: Path) -> dict[str, dict[str, set[str]]]:
    qrels: dict[str, dict[str, set[str]]] = {}
    for threshold in THRESHOLDS:
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in read_jsonl(processed / "qrels" / f"multi_{threshold}.jsonl"):
            grouped[row["qid"]].add(row["chunk_id"])
        qrels[threshold] = dict(grouped)
    return qrels


def build_multigold(
    ranking_paths: dict[int, Path], qrels: dict[str, dict[str, set[str]]]
) -> tuple[list[dict], list[dict]]:
    accumulators: dict[tuple[int, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for seed, path in ranking_paths.items():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for threshold in THRESHOLDS:
                    relevant = qrels[threshold].get(row["qid"])
                    if relevant is None:
                        continue
                    values = (
                        average_precision(row["ranking"], relevant),
                        binary_ndcg_at_k(row["ranking"], relevant, 10),
                    )
                    accumulators[(seed, row["method"], row["scenario"], threshold)].append(values)

    per_seed: list[dict] = []
    for seed in ranking_paths:
        for threshold in THRESHOLDS:
            for method in METHODS:
                scenario_rows = []
                for scenario in SCENARIOS:
                    values = np.asarray(accumulators[(seed, method, scenario, threshold)], dtype=float)
                    if values.size == 0:
                        raise AssertionError(f"Missing multi-gold rows: {seed}, {method}, {scenario}, {threshold}")
                    result = {
                        "seed": seed,
                        "threshold": threshold,
                        "method": method,
                        "scenario": scenario,
                        "n": len(values),
                        "map_50": float(values[:, 0].mean()),
                        "ndcg_10": float(values[:, 1].mean()),
                    }
                    per_seed.append(result)
                    scenario_rows.append(result)
                per_seed.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "method": method,
                        "scenario": "average",
                        "n": sum(row["n"] for row in scenario_rows),
                        "map_50": float(np.mean([row["map_50"] for row in scenario_rows])),
                        "ndcg_10": float(np.mean([row["ndcg_10"] for row in scenario_rows])),
                    }
                )

    summary: list[dict] = []
    for threshold in THRESHOLDS:
        for method in METHODS:
            for scenario in (*SCENARIOS, "average"):
                selected = [
                    row for row in per_seed
                    if row["threshold"] == threshold and row["method"] == method and row["scenario"] == scenario
                ]
                map_mean, map_sd = mean_sd(row["map_50"] for row in selected)
                ndcg_mean, ndcg_sd = mean_sd(row["ndcg_10"] for row in selected)
                summary.append(
                    {
                        "threshold": threshold,
                        "method": method,
                        "scenario": scenario,
                        "seeds": len(selected),
                        "n_total": sum(int(row["n"]) for row in selected),
                        "map_50_mean": map_mean,
                        "map_50_sd": map_sd,
                        "ndcg_10_mean": ndcg_mean,
                        "ndcg_10_sd": ndcg_sd,
                    }
                )
    return per_seed, summary


def build_robustness(query_rows: dict[int, list[dict[str, str]]]) -> tuple[list[dict], list[dict]]:
    per_seed: list[dict] = []
    for seed, rows in query_rows.items():
        for method in METHODS:
            for scenario in (*SCENARIOS, "average"):
                selected = rows if scenario == "average" else [row for row in rows if row["scenario"] == scenario]
                counts = robustness_counts(
                    [int(row[f"rank__{method}"]) for row in selected],
                    [int(row["rank__B1"]) for row in selected],
                )
                per_seed.append({"seed": seed, "method": method, "scenario": scenario, **counts})

    summary: list[dict] = []
    for method in METHODS:
        for scenario in (*SCENARIOS, "average"):
            selected = [row for row in per_seed if row["method"] == method and row["scenario"] == scenario]
            wins = sum(int(row["wins"]) for row in selected)
            ties = sum(int(row["ties"]) for row in selected)
            losses = sum(int(row["losses"]) for row in selected)
            total = wins + ties + losses
            ri_mean, ri_sd = mean_sd(row["robustness_index"] for row in selected)
            summary.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seeds": len(selected),
                    "n_total": total,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "robustness_index_pooled": (wins - losses) / total,
                    "robustness_index_mean": ri_mean,
                    "robustness_index_sd": ri_sd,
                }
            )
    return per_seed, summary


def wilcoxon_result(differences: np.ndarray) -> tuple[float, float]:
    if np.all(differences == 0):
        return 0.0, 1.0
    result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def stratified_bootstrap_ci(groups: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    bootstrap = np.zeros(BOOTSTRAP_RESAMPLES, dtype=float)
    for values in groups:
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
        bootstrap += values[indices].mean(axis=1) / len(groups)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return float(lower), float(upper)


def build_statistics(query_rows: dict[int, list[dict[str, str]]]) -> tuple[list[dict], list[dict]]:
    per_seed: list[dict] = []
    combined_differences: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    bootstrap_groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)

    for seed, rows in query_rows.items():
        seed_rows = []
        for left, right in COMPARISONS:
            differences = np.asarray(
                [reciprocal_rank(int(row[f"rank__{left}"])) - reciprocal_rank(int(row[f"rank__{right}"])) for row in rows],
                dtype=float,
            )
            statistic, p_value = wilcoxon_result(differences)
            result = {
                "seed": seed,
                "comparison": f"{left} vs {right}",
                "left": left,
                "right": right,
                "n": len(differences),
                "nonzero": int(np.count_nonzero(differences)),
                "delta_mrr": float(differences.mean()),
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "rank_biserial": rank_biserial_effect(differences),
            }
            seed_rows.append(result)
            combined_differences[(left, right)].append(differences)
            for scenario in SCENARIOS:
                scenario_differences = np.asarray(
                    [
                        reciprocal_rank(int(row[f"rank__{left}"]))
                        - reciprocal_rank(int(row[f"rank__{right}"]))
                        for row in rows if row["scenario"] == scenario
                    ],
                    dtype=float,
                )
                bootstrap_groups[(left, right)].append(scenario_differences)
        adjusted = holm_adjust([row["p_raw"] for row in seed_rows])
        for row, p_holm in zip(seed_rows, adjusted, strict=True):
            row["p_holm"] = float(p_holm)
            row["significant_raw_0.05"] = bool(row["p_raw"] < 0.05)
            row["significant_holm_0.05"] = bool(p_holm < 0.05)
            per_seed.append(row)

    combined: list[dict] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for left, right in COMPARISONS:
        differences = np.concatenate(combined_differences[(left, right)])
        statistic, p_value = wilcoxon_result(differences)
        ci_low, ci_high = stratified_bootstrap_ci(bootstrap_groups[(left, right)], rng)
        seed_selected = [row for row in per_seed if row["left"] == left and row["right"] == right]
        combined.append(
            {
                "comparison": f"{left} vs {right}",
                "left": left,
                "right": right,
                "n": len(differences),
                "nonzero": int(np.count_nonzero(differences)),
                "delta_mrr": float(np.mean([group.mean() for group in bootstrap_groups[(left, right)]])),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "rank_biserial": rank_biserial_effect(differences),
                "significant_seeds_raw": sum(bool(row["significant_raw_0.05"]) for row in seed_selected),
                "significant_seeds_holm": sum(bool(row["significant_holm_0.05"]) for row in seed_selected),
            }
        )
    adjusted = holm_adjust([row["p_raw"] for row in combined])
    for row, p_holm in zip(combined, adjusted, strict=True):
        row["p_holm"] = float(p_holm)
        row["significant_holm_0.05"] = bool(p_holm < 0.05)
    return per_seed, combined


def method_ordering(strict_summary: list[dict], multi_summary: list[dict]) -> dict:
    strict_values = {
        row["method"]: row["mrr_mean"]
        for row in strict_summary if row["scenario"] == "average"
    }
    strict_order = sorted(METHODS, key=lambda method: (-strict_values[method], METHODS.index(method)))
    strict_ranks = rankdata([-strict_values[method] for method in METHODS], method="average")
    multi = {}
    for threshold in THRESHOLDS:
        multi[threshold] = {}
        for metric in ("map_50", "ndcg_10"):
            values = {
                row["method"]: row[f"{metric}_mean"]
                for row in multi_summary
                if row["threshold"] == threshold and row["scenario"] == "average"
            }
            order = sorted(METHODS, key=lambda method: (-values[method], METHODS.index(method)))
            ranks = rankdata([-values[method] for method in METHODS], method="average")
            correlation = spearmanr(strict_ranks, ranks).statistic
            multi[threshold][metric] = {
                "ordering": order,
                "spearman_vs_strict_mrr": float(correlation),
                "top_method": order[0],
            }
    return {
        "strict_mrr": {"ordering": strict_order, "top_method": strict_order[0]},
        "multi_gold": multi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    verification = json.loads((args.results / "stage7" / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or verification.get("test_used_for_tuning") is not False:
        raise AssertionError("Stage 7 verification must pass before Stage 8")

    query_rows: dict[int, list[dict[str, str]]] = {}
    ranking_paths: dict[int, Path] = {}
    for seed in range(1, 11):
        query_path = args.results / "runs" / f"seed_{seed:02d}_queries.csv.gz"
        ranking_path = args.results / "runs" / f"seed_{seed:02d}_rankings.jsonl.gz"
        query_rows[seed] = read_query_rows(query_path)
        ranking_paths[seed] = ranking_path

    output = args.results / "stage8"
    strict_per_seed, strict_summary = build_strict(query_rows)
    multi_per_seed, multi_summary = build_multigold(ranking_paths, load_multi_qrels(args.processed))
    robustness_per_seed, robustness_summary = build_robustness(query_rows)
    statistics_per_seed, statistics_combined = build_statistics(query_rows)
    ordering = method_ordering(strict_summary, multi_summary)

    files_and_rows = (
        ("strict_per_seed.csv", strict_per_seed),
        ("strict_summary.csv", strict_summary),
        ("multigold_per_seed.csv", multi_per_seed),
        ("multigold_summary.csv", multi_summary),
        ("robustness_per_seed.csv", robustness_per_seed),
        ("robustness_summary.csv", robustness_summary),
        ("statistics_per_seed.csv", statistics_per_seed),
        ("statistics_combined.csv", statistics_combined),
    )
    for name, rows in files_and_rows:
        write_csv(output / name, rows)
    write_json(output / "method_ordering.json", ordering)

    expected_queries = sum(len(rows) for rows in query_rows.values())
    if expected_queries != verification["test_queries"]:
        raise AssertionError("Stage 8 query count differs from verified Stage 7 count")
    audit = {
        "status": "PASS",
        "source_stage7_status": verification["status"],
        "test_queries": expected_queries,
        "seeds": list(query_rows),
        "methods": list(METHODS),
        "scenarios": list(SCENARIOS),
        "strict_metrics": list(STRICT_METRICS),
        "multi_gold_thresholds": list(THRESHOLDS),
        "multi_gold_ap_cutoff": 50,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_comparisons": [f"{left} vs {right}" for left, right in COMPARISONS],
        "holm_family": "seven comparisons, separately within each seed and for the combined analysis",
        "outputs": {
            path.name: {"rows": len(rows), "sha256": file_sha256(path)}
            for name, rows in files_and_rows
            for path in [output / name]
        },
    }
    audit["outputs"]["method_ordering.json"] = {
        "sha256": file_sha256(output / "method_ordering.json")
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
