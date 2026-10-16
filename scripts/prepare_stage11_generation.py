"""Prepare private Stage 11 generation inputs from the frozen seed-1 runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("B1", "RM3", "SOFT", "FINAL", "ORACLE")
SCENARIOS = ("TPL", "PARA")


def read_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("rt", encoding="utf-8")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output_path: Path) -> dict:
    qa_rows = read_jsonl(ROOT / "data" / "processed" / "qa.jsonl")
    chunk_rows = read_jsonl(ROOT / "data" / "processed" / "chunks.jsonl")
    ranking_rows = read_jsonl(
        ROOT / "results" / "runs" / "seed_01_rankings.jsonl.gz"
    )

    qa_by_id = {row["qid"]: row for row in qa_rows}
    chunks = {row["chunk_id"]: row["text"] for row in chunk_rows}
    wanted = {
        (row["qid"], row["scenario"], row["method"]): row
        for row in ranking_rows
        if row["partition"] == "test"
        and row["scenario"] in SCENARIOS
        and row["method"] in CONFIGS[:-1]
    }

    observations = sorted(
        {(qid, scenario) for qid, scenario, _ in wanted},
        key=lambda item: (item[1], item[0]),
    )
    records: list[dict] = []
    for qid, scenario in observations:
        qa = qa_by_id[qid]
        for config in CONFIGS:
            if config == "ORACLE":
                chunk_ids = [qa["gold_chunk_id"]]
            else:
                rank = wanted[(qid, scenario, config)]
                if rank["gold_chunk_id"] != qa["gold_chunk_id"]:
                    raise ValueError(f"Gold mismatch for {qid}/{scenario}/{config}")
                chunk_ids = rank["ranking"][:3]

            texts = [chunks[chunk_id] for chunk_id in chunk_ids]
            texts.extend([""] * (3 - len(texts)))
            ids = chunk_ids + [None] * (3 - len(chunk_ids))
            records.append(
                {
                    "id": f"{qid}::{scenario}",
                    "qid": qid,
                    "scenario": scenario,
                    "configuration": config,
                    "question": qa["question"],
                    "chunk_ids": ids,
                    "chunk_1": texts[0],
                    "chunk_2": texts[1],
                    "chunk_3": texts[2],
                    "references": qa["answer_texts"],
                    "gold_chunk_id": qa["gold_chunk_id"],
                }
            )

    expected_observations = 173 * len(SCENARIOS)
    expected_rows = expected_observations * len(CONFIGS)
    if len(observations) != expected_observations or len(records) != expected_rows:
        raise ValueError(
            f"Unexpected coverage: {len(observations)} observations, {len(records)} rows"
        )
    counts = Counter((row["scenario"], row["configuration"]) for row in records)
    if set(counts.values()) != {173}:
        raise ValueError(f"Unbalanced coverage: {counts}")
    if len({(row["id"], row["configuration"]) for row in records}) != len(records):
        raise ValueError("Duplicate cache keys found")
    for row in records:
        if row["configuration"] == "ORACLE":
            assert row["chunk_ids"] == [row["gold_chunk_id"], None, None]
        else:
            assert all(row[f"chunk_{i}"] for i in (1, 2, 3))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "path": str(output_path),
        "rows": len(records),
        "observations": len(observations),
        "qids": len({qid for qid, _ in observations}),
        "counts": {f"{scenario}/{config}": count for (scenario, config), count in sorted(counts.items())},
        "sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "kaggle_io" / "gen_inputs.jsonl",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
