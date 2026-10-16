"""Verify Kaggle reranker output and install the local score cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import write_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-dir", type=Path, default=Path("kaggle_io/stage6_reranker"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/reranker"))
    args = parser.parse_args()
    expected = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    scores_path = args.output_dir / "scores.jsonl"
    if manifest["input_sha256"] != expected["input_sha256"]:
        raise ValueError("Hash input cache reranker tidak cocok")
    if digest(scores_path) != manifest["scores_sha256"]:
        raise ValueError("Hash scores.jsonl tidak cocok")
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line]
    scores = {row["pair_id"]: float(row["score"]) for row in rows}
    if len(scores) != expected["unique_pairs"] or len(scores) != len(rows):
        raise ValueError("Jumlah skor reranker tidak lengkap atau mengandung duplikat")
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Cache reranker memuat skor non-finite")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.cache_dir / "stage6_scores.json", scores)
    write_json(args.cache_dir / "stage6_manifest.json", manifest)
    print(json.dumps({
        "status": "PASS", "scores": len(scores), "model_id": manifest["model_id"],
        "revision": manifest["revision"], "fallback_used": manifest["fallback_used"],
    }))


if __name__ == "__main__":
    main()
