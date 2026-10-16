"""Verify final reports, paper artifacts, figures, and reproducibility entry points."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = [
        "reports/FINAL_REPORT.md",
        "reports/DRAFT_BAB3_METODOLOGI.md",
        "reports/DRAFT_BAB4_HASIL.md",
        "paper/outline.md",
        "paper/limitations_and_ethics.md",
        "paper/reproducibility.md",
        "paper/tables/main_results.csv",
        "paper/tables/main_results.tex",
        "paper/tables/statistical_comparisons.csv",
        "paper/tables/statistical_comparisons.tex",
        "paper/tables/component_decomposition.csv",
        "paper/tables/component_decomposition.tex",
        "paper/figures/main_mrr_comparison.png",
        "paper/figures/main_mrr_comparison.pdf",
        "scripts/run_all.ps1",
        "scripts/run_all.sh",
        "README.md",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    assert not missing, missing

    with (ROOT / "paper" / "tables" / "main_results.csv").open(encoding="utf-8") as handle:
        main_rows = {row["Method"]: row for row in csv.DictReader(handle)}
    assert main_rows["B1"]["Mean MRR"] == "0.2478"
    assert main_rows["FINAL"]["Mean MRR"] == "0.3961"
    assert main_rows["FINAL"]["R@10"] == "0.7206"
    assert main_rows["FINAL"]["nDCG@10"] == "0.4650"

    for png in (ROOT / "paper" / "figures").glob("*.png"):
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert min(dpi) >= 299, (png, dpi)
            assert image.width >= 1000 and image.height >= 700, (png, image.size)
        assert png.with_suffix(".pdf").is_file()

    manifest = json.loads((ROOT / "paper" / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        assert sha256(ROOT / relative) == expected, relative

    for report in ("FINAL_REPORT.md", "DRAFT_BAB3_METODOLOGI.md", "DRAFT_BAB4_HASIL.md"):
        text = (ROOT / "reports" / report).read_text(encoding="utf-8")
        assert len(text) > 5000, report
    final_report = (ROOT / "reports" / "FINAL_REPORT.md").read_text(encoding="utf-8")
    for marker in ("0,3961", "[0,1405; 0,1559]", "SKIPPED", "93,97%"):
        assert marker in final_report
    print("Stage 13 verification: PASS")


if __name__ == "__main__":
    main()
