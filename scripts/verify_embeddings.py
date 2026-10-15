"""Verify the downloaded Stage 3 embedding cache against its manifest and source IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(processed: Path, embeddings: Path) -> dict:
    manifest = json.loads((embeddings / "manifest.json").read_text(encoding="utf-8"))
    ids = json.loads((embeddings / "ids.json").read_text(encoding="utf-8"))
    chunks = read_jsonl(processed / "chunks.jsonl")
    qa = read_jsonl(processed / "qa.jsonl")
    expected_rows = {"chunks.npy": len(chunks), "questions.npy": len(qa), "answers.npy": len(qa)}
    files = {}
    for filename, rows in expected_rows.items():
        values = np.load(embeddings / filename, mmap_mode="r")
        norms = np.linalg.norm(values, axis=1)
        digest = sha256(embeddings / filename)
        files[filename] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite": bool(np.isfinite(values).all()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "shape_ok": values.shape == (rows, int(manifest["dimension"])),
            "hash_ok": digest == manifest["sha256"][filename],
        }
    ids_result = {
        "chunk_ids_match": ids["chunk_ids"] == [row["chunk_id"] for row in chunks],
        "qids_match": ids["qids"] == [row["qid"] for row in qa],
        "unique_chunk_ids": len(set(ids["chunk_ids"])) == len(ids["chunk_ids"]),
        "unique_qids": len(set(ids["qids"])) == len(ids["qids"]),
        "hash_ok": sha256(embeddings / "ids.json") == manifest["sha256"]["ids.json"],
    }
    normalized = all(
        row["norm_min"] >= 1.0 - 1e-5 and row["norm_max"] <= 1.0 + 1e-5
        for row in files.values()
    )
    ok = all(
        row["finite"] and row["shape_ok"] and row["hash_ok"] for row in files.values()
    ) and all(ids_result.values()) and normalized
    return {"status": "PASS" if ok else "FAIL", "files": files, "ids": ids_result, "normalized": normalized}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--output", type=Path, default=Path("results/embeddings/verification.json"))
    args = parser.parse_args()
    result = verify(args.processed, args.embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
