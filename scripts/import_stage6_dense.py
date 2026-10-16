"""Verify and expand Kaggle Stage 6 query embeddings into the local cache."""

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

from src.dense import validate_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    local = Path("kaggle_io/stage6_dense")
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((args.output_dir / "hashes.json").read_text(encoding="utf-8"))
    values_path = args.output_dir / "vectors.npy"
    if hashlib.sha256(values_path.read_bytes()).hexdigest() != manifest["sha256"]["vectors.npy"]:
        raise ValueError("Hash vectors.npy tidak cocok")
    values = validate_embeddings(np.load(values_path), rows=len(hashes))
    sequence = json.loads((local / "stage6_dense_sequence.json").read_text(encoding="utf-8"))
    lookup = {digest: values[index] for index, digest in enumerate(hashes)}
    if set(sequence) - set(lookup):
        raise ValueError("Output Kaggle tidak mencakup seluruh input")
    expanded = np.asarray([lookup[digest] for digest in sequence], dtype=np.float32)
    embeddings = Path("models/embeddings")
    np.save(embeddings / "stage6_full_dev.npy", expanded)
    (embeddings / "stage6_full_dev_ids.json").write_text(
        json.dumps(sequence, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "unique": len(hashes), "expanded": len(sequence)}))


if __name__ == "__main__":
    main()
