"""Run question-type, weighting, oracle-gap, and sampled failure analyses."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage7 import SCENARIOS, file_sha256
from src.data import QUESTION_TYPE_ORDER, read_jsonl, write_json
from src.metrics import reciprocal_rank, robustness_counts


METHODS = ("B1", "SOFT", "FINAL")
FAILURE_SAMPLE_SIZE = 20
FAILURE_SAMPLE_SEED = 20261016


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


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def analyze_question_types(query_rows: dict[int, list[dict[str, str]]]) -> tuple[list[dict], list[dict]]:
    per_seed = []
    for seed, rows in query_rows.items():
        for question_type in QUESTION_TYPE_ORDER:
            selected = [row for row in rows if row["question_type"] == question_type]
            baseline = [int(row["rank__B1"]) for row in selected]
            for method in METHODS:
                ranks = [int(row[f"rank__{method}"]) for row in selected]
                robustness = robustness_counts(ranks, baseline)
                per_seed.append(
                    {
                        "seed": seed,
                        "question_type": question_type,
                        "method": method,
                        "n": len(selected),
                        "mrr": float(np.mean([reciprocal_rank(rank) for rank in ranks])),
                        **robustness,
                    }
                )

    summary = []
    for question_type in QUESTION_TYPE_ORDER:
        for method in METHODS:
            selected = [
                row for row in per_seed
                if row["question_type"] == question_type and row["method"] == method
            ]
            mrr_mean, mrr_sd = mean_sd(row["mrr"] for row in selected)
            ri_mean, ri_sd = mean_sd(row["robustness_index"] for row in selected)
            wins = sum(int(row["wins"]) for row in selected)
            ties = sum(int(row["ties"]) for row in selected)
            losses = sum(int(row["losses"]) for row in selected)
            total = wins + ties + losses
            summary.append(
                {
                    "question_type": question_type,
                    "method": method,
                    "seeds": len(selected),
                    "n_total": total,
                    "mrr_mean": mrr_mean,
                    "mrr_sd": mrr_sd,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "robustness_index_pooled": (wins - losses) / total,
                    "robustness_index_mean": ri_mean,
                    "robustness_index_sd": ri_sd,
                }
            )
    return per_seed, summary


def analyze_weights(query_rows: dict[int, list[dict[str, str]]]) -> tuple[list[dict], list[dict]]:
    per_seed = []
    for seed, rows in query_rows.items():
        for scenario in SCENARIOS:
            values = np.asarray([float(row["w"]) for row in rows if row["scenario"] == scenario])
            per_seed.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "q25": float(np.percentile(values, 25)),
                    "median": float(np.median(values)),
                    "q75": float(np.percentile(values, 75)),
                    "max": float(values.max()),
                }
            )
    summary = []
    all_rows = [row for rows in query_rows.values() for row in rows]
    for scenario in SCENARIOS:
        pooled = np.asarray([float(row["w"]) for row in all_rows if row["scenario"] == scenario])
        selected = [row for row in per_seed if row["scenario"] == scenario]
        seed_mean, seed_sd = mean_sd(row["mean"] for row in selected)
        summary.append(
            {
                "scenario": scenario,
                "n_total": len(pooled),
                "pooled_min": float(pooled.min()),
                "pooled_q25": float(np.percentile(pooled, 25)),
                "pooled_median": float(np.median(pooled)),
                "pooled_q75": float(np.percentile(pooled, 75)),
                "pooled_max": float(pooled.max()),
                "seed_mean": seed_mean,
                "seed_mean_sd": seed_sd,
            }
        )
    return per_seed, summary


def analyze_oracle_gap(query_rows: dict[int, list[dict[str, str]]]) -> tuple[list[dict], list[dict]]:
    per_seed = []
    for seed, rows in query_rows.items():
        scenario_rows = []
        for scenario in SCENARIOS:
            selected = [row for row in rows if row["scenario"] == scenario]
            soft = float(np.mean([reciprocal_rank(int(row["rank__SOFT"])) for row in selected]))
            oracle = float(np.mean([reciprocal_rank(int(row["rank__ORACLE-W"])) for row in selected]))
            result = {
                "seed": seed,
                "scenario": scenario,
                "n": len(selected),
                "soft_mrr": soft,
                "oracle_mrr": oracle,
                "oracle_minus_soft": oracle - soft,
            }
            per_seed.append(result)
            scenario_rows.append(result)
        per_seed.append(
            {
                "seed": seed,
                "scenario": "average",
                "n": sum(row["n"] for row in scenario_rows),
                **{
                    key: float(np.mean([row[key] for row in scenario_rows]))
                    for key in ("soft_mrr", "oracle_mrr", "oracle_minus_soft")
                },
            }
        )
    summary = []
    for scenario in (*SCENARIOS, "average"):
        selected = [row for row in per_seed if row["scenario"] == scenario]
        result = {"scenario": scenario, "seeds": len(selected), "n_total": sum(row["n"] for row in selected)}
        for key in ("soft_mrr", "oracle_mrr", "oracle_minus_soft"):
            result[f"{key}_mean"], result[f"{key}_sd"] = mean_sd(row[key] for row in selected)
        summary.append(result)
    return per_seed, summary


def table_like(text: str) -> bool:
    """Detect a markdown-style table without exposing clinical text."""

    return bool(re.search(r"\|\s*:?-{3,}", text))


def analyze_failures(
    query_rows: dict[int, list[dict[str, str]]], processed: Path,
) -> tuple[list[dict], dict]:
    rows = [row for seed_rows in query_rows.values() for row in seed_rows]
    failed = [row for row in rows if int(row["rank__FINAL"]) > 10]
    by_qid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in failed:
        by_qid[row["qid"]].append(row)
    if len(by_qid) < FAILURE_SAMPLE_SIZE:
        raise AssertionError("Not enough distinct failed QA for the requested sample")

    rng = random.Random(FAILURE_SAMPLE_SEED)
    sampled_qids = rng.sample(sorted(by_qid), FAILURE_SAMPLE_SIZE)
    selected = [rng.choice(sorted(by_qid[qid], key=lambda row: (int(row["seed"]), row["scenario"]))) for qid in sampled_qids]

    chunks = {row["chunk_id"]: row for row in read_jsonl(processed / "chunks.jsonl")}
    qa = {row["qid"]: row for row in read_jsonl(processed / "qa.jsonl")}
    multi_counts = Counter(row["qid"] for row in read_jsonl(processed / "qrels" / "multi_0.7.jsonl"))
    result = []
    for row in selected:
        gold_text = chunks[qa[row["qid"]]["gold_chunk_id"]]["text"]
        wrong_type = row["scenario"] in {"TPL", "PARA"} and int(row["feedback_type_matches"]) == 0
        gold_table = table_like(gold_text)
        answer_spread = multi_counts[row["qid"]] > 1
        paraphrase_missed = row["scenario"] in {"PARA", "PNEW"} and float(row["w"]) <= 0.25
        if wrong_type:
            category = "feedback_salah_tipe"
        elif gold_table:
            category = "chunk_gold_tabel"
        elif answer_spread:
            category = "jawaban_tersebar"
        elif paraphrase_missed:
            category = "parafrasa_tidak_dikenali"
        else:
            category = "lainnya"
        result.append(
            {
                "sample_index": len(result) + 1,
                "seed": int(row["seed"]),
                "qid": row["qid"],
                "scenario": row["scenario"],
                "question_type": row["question_type"],
                "rank_b1": int(row["rank__B1"]),
                "rank_soft": int(row["rank__SOFT"]),
                "rank_final": int(row["rank__FINAL"]),
                "s": float(row["s"]),
                "w": float(row["w"]),
                "feedback_type_matches": int(row["feedback_type_matches"]),
                "feedback_salah_tipe": wrong_type,
                "chunk_gold_tabel": gold_table,
                "multi_gold_0.7_chunks": multi_counts[row["qid"]],
                "jawaban_tersebar": answer_spread,
                "parafrasa_tidak_dikenali": paraphrase_missed,
                "primary_category": category,
            }
        )
    categories = Counter(row["primary_category"] for row in result)
    flags = {
        key: sum(bool(row[key]) for row in result)
        for key in (
            "feedback_salah_tipe",
            "chunk_gold_tabel",
            "jawaban_tersebar",
            "parafrasa_tidak_dikenali",
        )
    }
    summary = {
        "failure_definition": "FINAL strict-gold rank > 10",
        "sample_unit": "20 distinct QA, one failed seed-scenario observation sampled per QA",
        "sample_seed": FAILURE_SAMPLE_SEED,
        "total_observations": len(rows),
        "failed_observations": len(failed),
        "failure_rate": len(failed) / len(rows),
        "distinct_failed_qids": len(by_qid),
        "sample_size": len(result),
        "primary_categories": dict(sorted(categories.items())),
        "nonexclusive_flags": flags,
        "contains_clinical_text": False,
    }
    return result, summary


def plot_question_types(summary: list[dict], path: Path) -> None:
    lookup = {(row["question_type"], row["method"]): row for row in summary}
    x = np.arange(len(QUESTION_TYPE_ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    width = 0.25
    for offset, method, color in zip((-width, 0, width), METHODS, ("#9C9C9C", "#4C78A8", "#E45756"), strict=True):
        values = [float(lookup[(question_type, method)]["mrr_mean"]) for question_type in QUESTION_TYPE_ORDER]
        errors = [float(lookup[(question_type, method)]["mrr_sd"]) for question_type in QUESTION_TYPE_ORDER]
        axes[0].bar(x + offset, values, width, yerr=errors, capsize=2, label=method, color=color)
    axes[0].set_ylabel("MRR mean +/- SD")
    axes[0].set_title("MRR strict per tipe pertanyaan")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for offset, method, color in zip((-width / 2, width / 2), ("SOFT", "FINAL"), ("#4C78A8", "#E45756"), strict=True):
        values = [float(lookup[(question_type, method)]["robustness_index_mean"]) for question_type in QUESTION_TYPE_ORDER]
        errors = [float(lookup[(question_type, method)]["robustness_index_sd"]) for question_type in QUESTION_TYPE_ORDER]
        axes[1].bar(x + offset, values, width, yerr=errors, capsize=2, label=method, color=color)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Robustness Index mean +/- SD")
    axes[1].set_title("Ketahanan terhadap B1 per tipe pertanyaan")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, QUESTION_TYPE_ORDER, rotation=25, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weights(query_rows: dict[int, list[dict[str, str]]], path: Path) -> None:
    rows = [row for seed_rows in query_rows.values() for row in seed_rows]
    values = [[float(row["w"]) for row in rows if row["scenario"] == scenario] for scenario in SCENARIOS]
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    parts = axis.violinplot(values, showmeans=True, showmedians=True, showextrema=True)
    for body, color in zip(parts["bodies"], ("#4C78A8", "#F58518", "#54A24B", "#E45756"), strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    axis.set_xticks(range(1, len(SCENARIOS) + 1), SCENARIOS)
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Bobot feedback QA w(s)")
    axis.set_title("Distribusi bobot adaptif pada test, 10 seed")
    axis.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    started = time.perf_counter()
    stage9 = json.loads((args.results / "stage9" / "verification.json").read_text(encoding="utf-8"))
    if stage9.get("status") != "PASS" or stage9.get("test_used_for_selection") is not False:
        raise AssertionError("Stage 9 verification must pass before Stage 10")

    query_rows = {
        seed: read_query_rows(args.results / "runs" / f"seed_{seed:02d}_queries.csv.gz")
        for seed in range(1, 11)
    }
    output = args.results / "stage10"
    type_per_seed, type_summary = analyze_question_types(query_rows)
    weight_per_seed, weight_summary = analyze_weights(query_rows)
    gap_per_seed, gap_summary = analyze_oracle_gap(query_rows)
    failures, failure_summary = analyze_failures(query_rows, args.processed)

    csv_outputs = (
        ("question_type_per_seed.csv", type_per_seed),
        ("question_type_summary.csv", type_summary),
        ("weight_per_seed.csv", weight_per_seed),
        ("weight_summary.csv", weight_summary),
        ("oracle_gap_per_seed.csv", gap_per_seed),
        ("oracle_gap_summary.csv", gap_summary),
        ("failure_sample.csv", failures),
    )
    for name, rows in csv_outputs:
        write_csv(output / name, rows)
    write_json(output / "failure_summary.json", failure_summary)

    type_figure = args.figures / "10_question_type_performance.png"
    weight_figure = args.figures / "10_weight_distribution.png"
    plot_question_types(type_summary, type_figure)
    plot_weights(query_rows, weight_figure)

    output_files = [output / name for name, _ in csv_outputs] + [
        output / "failure_summary.json", type_figure, weight_figure,
    ]
    audit = {
        "status": "PASS",
        "source_stage9_status": stage9["status"],
        "test_used_for_selection": False,
        "seeds": list(query_rows),
        "test_observations": sum(len(rows) for rows in query_rows.values()),
        "question_types": list(QUESTION_TYPE_ORDER),
        "methods": list(METHODS),
        "failure_sample_seed": FAILURE_SAMPLE_SEED,
        "failure_sample_size": FAILURE_SAMPLE_SIZE,
        "contains_clinical_text": False,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in output_files
        },
    }
    write_json(output / "audit.json", audit)
    print(json.dumps({**audit, "failure_summary": failure_summary}, indent=2))


if __name__ == "__main__":
    main()
