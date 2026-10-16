"""Verify Stage 12 trace coverage and arithmetic invariants."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    trace_path = ROOT / "results" / "stage12" / "manual_trace.json"
    report_path = ROOT / "reports" / "12_manual_trace.md"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    selection = trace["selection"]
    assert selection["available_chunks"] < 40
    assert selection["rank_soft"] < selection["rank_b1"]
    assert (selection["seed"], selection["qid"], selection["scenario"]) == (7, "Q0100", "TPL")
    assert len(trace["question_similarity"]["top3"]) == 3
    assert len(trace["feedback"]["qa"]) == trace["feedback"]["m"] == 5
    assert len(trace["feedback"]["external"]) == 5
    assert len(trace["baseline"]["top3"]) == 3
    assert len(trace["normalization"]["rows"]) == 3
    assert len(trace["reranker"]["top3"]) == 3

    weight = trace["soft"]["weight"]
    for row in trace["soft"]["rows"]:
        expected = weight * row["qa_normalized"] + (1.0 - weight) * row["ext_normalized"]
        assert math.isclose(expected, row["soft_score"], abs_tol=1e-12)
    for row in trace["reranker"]["top3"]:
        expected = 0.25 * row["reranker_normalized"] + 0.75 * row["pipeline_candidate_normalized"]
        assert math.isclose(expected, row["combined_score"], abs_tol=1e-12)

    ri = trace["robustness_example"]
    assert ri["wins"] + ri["ties"] + ri["losses"] == ri["n"]
    assert math.isclose(ri["ri"], (ri["wins"] - ri["losses"]) / ri["n"], abs_tol=1e-15)
    for heading in range(1, 10):
        assert f"## {heading}." in report
    for marker in ("rank **14**", "rank **1**", "0,379065", "Q0100"):
        assert marker in report
    print("Stage 12 verification: PASS")


if __name__ == "__main__":
    main()
