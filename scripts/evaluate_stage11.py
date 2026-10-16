"""Validate Kaggle generation outputs and write the Stage 11 report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("B1", "RM3", "SOFT", "FINAL", "ORACLE")
SCENARIOS = ("TPL", "PARA")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float) -> str:
    return f"{value:.4f}".replace(".", ",")


def paired_test(rows: list[dict], left: str, right: str) -> dict:
    keyed = {(row["id"], row["configuration"]): row for row in rows}
    ids = sorted({row["id"] for row in rows})
    left_values = np.array([keyed[(item, left)]["rouge_l_f1"] for item in ids])
    right_values = np.array([keyed[(item, right)]["rouge_l_f1"] for item in ids])
    delta = left_values - right_values
    if np.allclose(delta, 0.0):
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(delta, zero_method="pratt", alternative="two-sided")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {
        "comparison": f"{left} vs {right}",
        "n": len(ids),
        "mean_delta_rouge_l_f1": float(delta.mean()),
        "statistic": statistic,
        "pvalue": pvalue,
    }


def evaluate(outputs_path: Path, metrics_path: Path, report_path: Path) -> None:
    outputs = read_jsonl(outputs_path)
    metrics = read_jsonl(metrics_path)
    expected = 173 * 2 * 5
    for label, rows in (("outputs", outputs), ("metrics", metrics)):
        if len(rows) != expected:
            raise ValueError(f"{label}: expected {expected} rows, found {len(rows)}")
        keys = {(row["id"], row["configuration"]) for row in rows}
        if len(keys) != expected:
            raise ValueError(f"{label}: duplicate or missing cache keys")

    counts = Counter((row["scenario"], row["configuration"]) for row in metrics)
    if set(counts.values()) != {173}:
        raise ValueError(f"Unbalanced metrics coverage: {counts}")
    if any(not row.get("answer", "").strip() for row in outputs):
        raise ValueError("At least one generated answer is empty")
    required_metrics = {"rouge_l_f1", "bertscore_f1", "abstained"}
    if any(not required_metrics.issubset(row) for row in metrics):
        raise ValueError("Metric rows are incomplete")

    summary = []
    for scenario in (*SCENARIOS, "ALL"):
        subset = metrics if scenario == "ALL" else [r for r in metrics if r["scenario"] == scenario]
        for config in CONFIGS:
            selected = [r for r in subset if r["configuration"] == config]
            summary.append(
                {
                    "scenario": scenario,
                    "configuration": config,
                    "n": len(selected),
                    "rouge_l_f1": float(np.mean([r["rouge_l_f1"] for r in selected])),
                    "bertscore_f1": float(np.mean([r["bertscore_f1"] for r in selected])),
                    "abstain_rate": float(np.mean([r["abstained"] for r in selected])),
                }
            )

    tests = [paired_test(metrics, "FINAL", "B1"), paired_test(metrics, "SOFT", "B1")]
    model_ids = sorted({row.get("model_id", "unknown") for row in outputs})
    revisions = sorted({str(row.get("model_revision", "unknown")) for row in outputs})
    avg_seconds = float(np.mean([row["generation_seconds"] for row in outputs]))
    avg_tokens = float(np.mean([row["generated_tokens"] for row in outputs]))

    lines = [
        "# Tahap 11 — Evaluasi Generation",
        "",
        "## Ringkasan pelaksanaan",
        "",
        f"Evaluasi memakai **{len(metrics):,}** jawaban dari 173 QA seed 1 pada skenario TPL dan PARA. "
        "B1, RM3, SOFT, dan FINAL memakai tiga chunk teratas masing-masing; ORACLE memakai hanya chunk gold "
        "(K2 dan K3 kosong).",
        "",
        f"- Model: `{', '.join(model_ids)}`",
        f"- Revision: `{', '.join(revisions)}`",
        "- Decoding: greedy, maksimum 256 token baru",
        f"- Rata-rata waktu teramortisasi: {avg_seconds:.3f} detik/jawaban",
        f"- Rata-rata token keluaran: {avg_tokens:.2f}",
        "- BERTScore: `bert-base-multilingual-cased`",
        "",
        "## Hasil",
        "",
        "| Skenario | Konfigurasi | n | ROUGE-L F1 | BERTScore F1 | Abstain rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario']} | {row['configuration']} | {row['n']} | "
            f"{fmt(row['rouge_l_f1'])} | {fmt(row['bertscore_f1'])} | {fmt(row['abstain_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Uji Wilcoxon berpasangan",
            "",
            "Uji dua sisi memakai ROUGE-L F1 per pasangan id yang sama (346 observasi TPL/PARA).",
            "",
            "| Perbandingan | n | Δ mean ROUGE-L F1 | Statistik W | p-value |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in tests:
        lines.append(
            f"| {row['comparison']} | {row['n']} | {fmt(row['mean_delta_rouge_l_f1'])} | "
            f"{row['statistic']:.1f} | {row['pvalue']:.6g} |"
        )
    all_rows = {row["configuration"]: row for row in summary if row["scenario"] == "ALL"}
    lines.extend(
        [
            "",
            "## Interpretasi",
            "",
            f"Secara agregat, FINAL berbeda {fmt(all_rows['FINAL']['rouge_l_f1'] - all_rows['B1']['rouge_l_f1'])} "
            "ROUGE-L F1 dari B1, sedangkan SOFT berbeda "
            f"{fmt(all_rows['SOFT']['rouge_l_f1'] - all_rows['B1']['rouge_l_f1'])}. "
            "ORACLE diperlakukan sebagai batas atas konteks karena hanya menerima chunk gold. "
            "BERTScore dan abstain rate digunakan sebagai ukuran pendukung; hasil tidak ditafsirkan sebagai "
            "penilaian klinis atau faktualitas manual.",
            "",
            "## Bukti reproduksibilitas",
            "",
            f"- `gen_outputs.jsonl`: `{sha256(outputs_path)}`",
            f"- `gen_metrics.jsonl`: `{sha256(metrics_path)}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(metrics), "summary": summary, "tests": tests}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "kaggle_io" / "gen_outputs.jsonl")
    parser.add_argument("--metrics", type=Path, default=ROOT / "kaggle_io" / "gen_metrics.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "11_generation.md")
    args = parser.parse_args()
    evaluate(args.outputs, args.metrics, args.report)


if __name__ == "__main__":
    main()
