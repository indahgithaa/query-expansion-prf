"""Verify Stage 9 component-analysis outputs and their source consistency."""

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

from scripts.run_stage7 import file_sha256
from scripts.run_stage9 import DECOMPOSITION, QS_METHODS, SCENARIOS
from src.data import write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    output = args.results / "stage9"
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["test_used_for_selection"] is not False:
        raise AssertionError("Stage 9 audit did not pass")
    for name, metadata in audit["outputs"].items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise AssertionError(f"Missing or resized output: {name}")
        if file_sha256(path) != metadata["sha256"]:
            raise AssertionError(f"Output hash mismatch: {name}")

    similarity = read_csv(output / "question_similarity.csv")
    if {row["qs_method"] for row in similarity} != set(QS_METHODS):
        raise AssertionError("Question-similarity coverage mismatch")
    stage6 = {
        row["qs_method"]: row
        for row in read_csv(args.results / "stage6" / "qs_candidates.csv")
    }
    for row in similarity:
        for metric in (*SCENARIOS, "mean"):
            value = float(row[metric])
            if not 0 <= value <= 1:
                raise AssertionError(f"Invalid MRR: {row['qs_method']}, {metric}")
            if row["qs_method"] in stage6 and not math.isclose(
                value, float(stage6[row["qs_method"]][metric]), abs_tol=1e-12,
            ):
                raise AssertionError(f"Stage 6 reproduction mismatch: {row['qs_method']}, {metric}")

    feedback = read_csv(output / "feedback_accuracy.csv")
    expected_feedback = {(method, scenario) for method in QS_METHODS for scenario in ("TPL", "PARA")}
    if {(row["method"], row["scenario"]) for row in feedback} != expected_feedback:
        raise AssertionError("Feedback-accuracy coverage mismatch")
    for row in feedback:
        matches, total = int(row["matching_feedback"]), int(row["feedback_items"])
        if total <= 0 or not math.isclose(float(row["accuracy"]), matches / total, abs_tol=1e-12):
            raise AssertionError(f"Feedback accuracy mismatch: {row}")

    scores = read_csv(output / "similarity_scores.csv")
    counts = {}
    for method in QS_METHODS:
        for scenario in SCENARIOS:
            count = sum(row["method"] == method and row["scenario"] == scenario for row in scores)
            counts[(method, scenario)] = count
    if set(counts.values()) != {513} or any(not math.isfinite(float(row["s"])) for row in scores):
        raise AssertionError("Similarity-score coverage or finiteness mismatch")

    retriever = read_csv(output / "retriever_analysis.csv")
    if len(retriever) != 5 or len({row["retriever"] for row in retriever}) != 5:
        raise AssertionError("Retriever summary coverage mismatch")
    for row in retriever:
        if any(not 0 <= float(row[metric]) <= 1 for metric in (*SCENARIOS, "mean")):
            raise AssertionError(f"Invalid retriever MRR: {row}")

    reranker = read_csv(output / "reranker_analysis.csv")
    expected_reranker = {
        (method, scenario)
        for method in ("B1", "B1+RR", "SOFT", "FINAL")
        for scenario in (*SCENARIOS, "average")
    }
    if {(row["method"], row["scenario"]) for row in reranker} != expected_reranker:
        raise AssertionError("Reranker summary coverage mismatch")

    decomposition = read_csv(output / "decomposition.csv")
    if [(row["stage"], row["method"]) for row in decomposition] != list(DECOMPOSITION):
        raise AssertionError("Decomposition order mismatch")
    gains = [float(row["delta_average_vs_previous"]) for row in decomposition[1:]]
    total = float(decomposition[-1]["average"]) - float(decomposition[0]["average"])
    if not math.isclose(sum(gains), total, abs_tol=1e-12):
        raise AssertionError("Decomposition does not sum to B1--FINAL gain")

    lambdas = json.loads((output / "lambda_selection.json").read_text(encoding="utf-8"))
    if sum(lambdas["B1+RR"].values()) != 10 or sum(lambdas["FINAL"].values()) != 10:
        raise AssertionError("Lambda selection does not cover ten seeds")

    result = {
        "status": "PASS",
        "hashes_verified": len(audit["outputs"]),
        "question_similarity_methods": len(similarity),
        "similarity_score_rows": len(scores),
        "feedback_accuracy_rows": len(feedback),
        "retriever_rows": len(retriever),
        "reranker_rows": len(reranker),
        "decomposition_rows": len(decomposition),
        "test_used_for_selection": False,
    }
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
