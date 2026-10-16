"""Verify completeness and internal consistency of Stage 8 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage7 import METHODS, SCENARIOS
from scripts.run_stage8 import COMPARISONS, STRICT_METRICS, THRESHOLDS, file_sha256
from src.data import write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_between(value: str, lower: float, upper: float) -> bool:
    number = float(value)
    return math.isfinite(number) and lower <= number <= upper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    output = args.results / "stage8"
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["source_stage7_status"] != "PASS":
        raise AssertionError("Stage 8 or source Stage 7 audit did not pass")

    for name, metadata in audit["outputs"].items():
        path = output / name
        if not path.is_file() or file_sha256(path) != metadata["sha256"]:
            raise AssertionError(f"Output hash mismatch: {name}")
        if "rows" in metadata and len(read_csv(path)) != metadata["rows"]:
            raise AssertionError(f"Output row count mismatch: {name}")

    strict = read_csv(output / "strict_summary.csv")
    expected_strict = {(method, scenario) for method in METHODS for scenario in (*SCENARIOS, "average")}
    if {(row["method"], row["scenario"]) for row in strict} != expected_strict:
        raise AssertionError("Strict summary coverage mismatch")
    for row in strict:
        if int(row["seeds"]) != 10:
            raise AssertionError("Strict summary must contain ten seeds")
        for metric in STRICT_METRICS:
            if not finite_between(row[f"{metric}_mean"], 0.0, 1.0) or float(row[f"{metric}_sd"]) < 0:
                raise AssertionError(f"Invalid strict metric: {row['method']}, {row['scenario']}, {metric}")

    multi = read_csv(output / "multigold_summary.csv")
    expected_multi = {
        (threshold, method, scenario)
        for threshold in THRESHOLDS for method in METHODS for scenario in (*SCENARIOS, "average")
    }
    if {(row["threshold"], row["method"], row["scenario"]) for row in multi} != expected_multi:
        raise AssertionError("Multi-gold summary coverage mismatch")
    for row in multi:
        for metric in ("map_50", "ndcg_10"):
            if not finite_between(row[f"{metric}_mean"], 0.0, 1.0) or float(row[f"{metric}_sd"]) < 0:
                raise AssertionError(f"Invalid multi-gold metric: {row}")

    robustness = read_csv(output / "robustness_summary.csv")
    if {(row["method"], row["scenario"]) for row in robustness} != expected_strict:
        raise AssertionError("Robustness summary coverage mismatch")
    for row in robustness:
        wins, ties, losses, total = (int(row[key]) for key in ("wins", "ties", "losses", "n_total"))
        if wins + ties + losses != total:
            raise AssertionError(f"Robustness counts do not sum: {row}")
        expected_ri = (wins - losses) / total
        if not math.isclose(float(row["robustness_index_pooled"]), expected_ri, abs_tol=1e-12):
            raise AssertionError(f"Robustness index mismatch: {row}")

    statistics = read_csv(output / "statistics_combined.csv")
    expected_comparisons = {f"{left} vs {right}" for left, right in COMPARISONS}
    if {row["comparison"] for row in statistics} != expected_comparisons:
        raise AssertionError("Statistical comparison coverage mismatch")
    for row in statistics:
        if int(row["bootstrap_resamples"]) != 10_000:
            raise AssertionError("Bootstrap must use 10,000 resamples")
        if not finite_between(row["p_raw"], 0.0, 1.0) or not finite_between(row["p_holm"], 0.0, 1.0):
            raise AssertionError(f"Invalid p-value: {row}")
        if float(row["bootstrap_ci_low"]) > float(row["bootstrap_ci_high"]):
            raise AssertionError(f"Invalid confidence interval: {row}")
        if not -1 <= float(row["rank_biserial"]) <= 1:
            raise AssertionError(f"Invalid effect size: {row}")

    ordering = json.loads((output / "method_ordering.json").read_text(encoding="utf-8"))
    if set(ordering["strict_mrr"]["ordering"]) != set(METHODS):
        raise AssertionError("Strict method ordering is incomplete")
    for threshold in THRESHOLDS:
        for metric in ("map_50", "ndcg_10"):
            item = ordering["multi_gold"][threshold][metric]
            if set(item["ordering"]) != set(METHODS) or not -1 <= item["spearman_vs_strict_mrr"] <= 1:
                raise AssertionError("Multi-gold ordering is invalid")

    result = {
        "status": "PASS",
        "hashes_verified": len(audit["outputs"]),
        "strict_summary_rows": len(strict),
        "multigold_summary_rows": len(multi),
        "robustness_summary_rows": len(robustness),
        "statistical_comparisons": len(statistics),
        "bootstrap_resamples": 10_000,
    }
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
