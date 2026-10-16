"""Build publication tables, figures, and a machine-readable Stage 13 manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_latex(path: Path, rows: list[dict], caption: str, label: str) -> None:
    columns = list(rows[0])
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row[column]) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def f4(value: str | float) -> str:
    return f"{float(value):.4f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    strict = read_csv(ROOT / "results" / "stage8" / "strict_summary.csv")
    strict_lookup = {(row["method"], row["scenario"]): row for row in strict}
    methods = ("B1", "RM3", "P-QA", "P-FUS", "ZONE3", "SOFT", "B1+RR", "FINAL", "ORACLE-W")
    main_rows = []
    for method in methods:
        avg = strict_lookup[(method, "average")]
        main_rows.append({
            "Method": method,
            "TPL MRR": f4(strict_lookup[(method, "TPL")]["mrr_mean"]),
            "NEW MRR": f4(strict_lookup[(method, "NEW")]["mrr_mean"]),
            "PARA MRR": f4(strict_lookup[(method, "PARA")]["mrr_mean"]),
            "PNEW MRR": f4(strict_lookup[(method, "PNEW")]["mrr_mean"]),
            "Mean MRR": f4(avg["mrr_mean"]),
            "R@10": f4(avg["recall_10_mean"]),
            "nDCG@10": f4(avg["ndcg_10_mean"]),
        })
    write_csv(TABLES / "main_results.csv", main_rows)
    write_latex(
        TABLES / "main_results.tex", main_rows,
        "Hasil retrieval gold ketat, rata-rata 10 pengulangan.", "tab:main-results",
    )

    stats = read_csv(ROOT / "results" / "stage8" / "statistics_combined.csv")
    stats_rows = [{
        "Comparison": row["comparison"],
        "Delta MRR": f4(row["delta_mrr"]),
        "CI low": f4(row["bootstrap_ci_low"]),
        "CI high": f4(row["bootstrap_ci_high"]),
        "Holm p": "<1e-300" if float(row["p_holm"]) == 0 else f"{float(row['p_holm']):.3e}",
        "Rank-biserial": f4(row["rank_biserial"]),
        "Significant seeds": f"{row['significant_seeds_holm']}/10",
    } for row in stats]
    write_csv(TABLES / "statistical_comparisons.csv", stats_rows)
    write_latex(
        TABLES / "statistical_comparisons.tex", stats_rows,
        "Uji statistik berpasangan pada reciprocal rank.", "tab:statistics",
    )

    decomposition = read_csv(ROOT / "results" / "stage9" / "decomposition.csv")
    total_delta = float(decomposition[-1]["average"]) - float(decomposition[0]["average"])
    decomposition_rows = []
    for row in decomposition:
        delta = row["delta_average_vs_previous"]
        decomposition_rows.append({
            "Stage": row["stage"],
            "Method": row["method"],
            "Mean MRR": f4(row["average"]),
            "Increment": "--" if not delta else f4(delta),
            "Percent total": "--" if not delta else f"{100 * float(delta) / total_delta:.2f}",
        })
    write_csv(TABLES / "component_decomposition.csv", decomposition_rows)
    write_latex(
        TABLES / "component_decomposition.tex", decomposition_rows,
        "Dekomposisi berurutan kenaikan B1 ke FINAL.", "tab:decomposition",
    )

    scenarios = ("TPL", "NEW", "PARA", "PNEW")
    plotted = ("B1", "RM3", "SOFT", "FINAL")
    x = np.arange(len(scenarios))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, method in enumerate(plotted):
        values = [float(strict_lookup[(method, scenario)]["mrr_mean"]) for scenario in scenarios]
        errors = [float(strict_lookup[(method, scenario)]["mrr_sd"]) for scenario in scenarios]
        ax.bar(x + (offset - 1.5) * width, values, width, yerr=errors, capsize=2, label=method)
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 0.72)
    ax.set_title("Kinerja retrieval per skenario (mean ± SD, 10 seed)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_mrr_comparison.png", dpi=300)
    fig.savefig(FIGURES / "main_mrr_comparison.pdf")
    plt.close(fig)

    source_figures = (
        "09_similarity_score_distribution.png",
        "10_question_type_performance.png",
        "10_weight_distribution.png",
    )
    for name in source_figures:
        source = ROOT / "reports" / "figures" / name
        target = FIGURES / name
        shutil.copyfile(source, target)
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            rgb.save(FIGURES / name.replace(".png", ".pdf"), "PDF", resolution=300.0)

    tracked = sorted(path for path in PAPER.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "generated_by": "scripts/build_stage13_artifacts.py",
        "source_tables": [
            "results/stage8/strict_summary.csv",
            "results/stage8/statistics_combined.csv",
            "results/stage9/decomposition.csv",
        ],
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked},
    }
    (PAPER / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tables": 6, "figure_pairs": 4, "manifest_files": len(tracked)}, indent=2))


if __name__ == "__main__":
    main()
