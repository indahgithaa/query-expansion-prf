"""Verify completeness, formulas, privacy, and source consistency of Stage 10."""

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

from scripts.run_stage10 import FAILURE_SAMPLE_SIZE, METHODS, SCENARIOS
from scripts.run_stage7 import file_sha256
from src.data import QUESTION_TYPE_ORDER, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    output = args.results / "stage10"
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["test_used_for_selection"] is not False:
        raise AssertionError("Stage 10 audit did not pass")
    if audit["contains_clinical_text"] is not False:
        raise AssertionError("Stage 10 output must not contain clinical text")
    for name, metadata in audit["outputs"].items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise AssertionError(f"Missing or resized output: {name}")
        if file_sha256(path) != metadata["sha256"]:
            raise AssertionError(f"Output hash mismatch: {name}")

    type_rows = read_csv(output / "question_type_summary.csv")
    expected_types = {(question_type, method) for question_type in QUESTION_TYPE_ORDER for method in METHODS}
    if {(row["question_type"], row["method"]) for row in type_rows} != expected_types:
        raise AssertionError("Question-type coverage mismatch")
    for row in type_rows:
        wins, ties, losses, total = (int(row[key]) for key in ("wins", "ties", "losses", "n_total"))
        if wins + ties + losses != total:
            raise AssertionError(f"Question-type W/T/L mismatch: {row}")
        if not math.isclose(
            float(row["robustness_index_pooled"]), (wins - losses) / total, abs_tol=1e-12,
        ):
            raise AssertionError(f"Question-type RI mismatch: {row}")
        if not 0 <= float(row["mrr_mean"]) <= 1:
            raise AssertionError(f"Question-type MRR out of bounds: {row}")

    weights = read_csv(output / "weight_summary.csv")
    if {row["scenario"] for row in weights} != set(SCENARIOS):
        raise AssertionError("Weight scenario coverage mismatch")
    for row in weights:
        ordered = [float(row[key]) for key in ("pooled_min", "pooled_q25", "pooled_median", "pooled_q75", "pooled_max")]
        if ordered != sorted(ordered) or not 0 <= ordered[0] <= ordered[-1] <= 1:
            raise AssertionError(f"Weight quantiles invalid: {row}")

    gaps = read_csv(output / "oracle_gap_summary.csv")
    if {row["scenario"] for row in gaps} != {*SCENARIOS, "average"}:
        raise AssertionError("Oracle-gap scenario coverage mismatch")
    for row in gaps:
        expected = float(row["oracle_mrr_mean"]) - float(row["soft_mrr_mean"])
        if not math.isclose(float(row["oracle_minus_soft_mean"]), expected, abs_tol=1e-12):
            raise AssertionError(f"Oracle-gap arithmetic mismatch: {row}")

    failures = read_csv(output / "failure_sample.csv")
    summary = json.loads((output / "failure_summary.json").read_text(encoding="utf-8"))
    forbidden = {"text", "question", "answer", "answer_texts", "context", "report_title", "source_file"}
    if len(failures) != FAILURE_SAMPLE_SIZE or len({row["qid"] for row in failures}) != FAILURE_SAMPLE_SIZE:
        raise AssertionError("Failure sample must contain 20 distinct QA")
    if forbidden & set(failures[0]):
        raise AssertionError("Failure sample exposes clinical or source text")
    if any(int(row["rank_final"]) <= 10 for row in failures):
        raise AssertionError("Failure sample contains a successful top-10 observation")
    if summary["contains_clinical_text"] is not False or summary["sample_size"] != FAILURE_SAMPLE_SIZE:
        raise AssertionError("Failure summary privacy or sample size mismatch")
    if sum(summary["primary_categories"].values()) != FAILURE_SAMPLE_SIZE:
        raise AssertionError("Primary failure categories do not sum to sample size")

    result = {
        "status": "PASS",
        "hashes_verified": len(audit["outputs"]),
        "question_type_rows": len(type_rows),
        "weight_scenarios": len(weights),
        "oracle_gap_rows": len(gaps),
        "failure_sample_size": len(failures),
        "distinct_failure_qids": len({row["qid"] for row in failures}),
        "contains_clinical_text": False,
        "test_used_for_selection": False,
    }
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
