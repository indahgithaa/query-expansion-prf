"""Build normalized Stage 2 data, splits, strict qrels, and multi-gold qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    create_split,
    load_raw,
    reconcile_and_process,
    validate_split,
    write_json,
    write_jsonl,
)
from src.labels import build_qrels


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset/dataset.json"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/data_preparation"))
    args = parser.parse_args()

    raw_qa, chunk_files = load_raw(args.dataset, args.chunks_dir)
    documents, chunks, qa, reconciliation = reconcile_and_process(raw_qa, chunk_files)
    strict, multi, label_audit = build_qrels(qa, chunks)

    files = {
        "documents": args.output_dir / "documents.jsonl",
        "chunks": args.output_dir / "chunks.jsonl",
        "qa": args.output_dir / "qa.jsonl",
        "strict_qrels": args.output_dir / "qrels" / "strict.jsonl",
    }
    write_jsonl(files["documents"], documents)
    write_jsonl(files["chunks"], chunks)
    write_jsonl(files["qa"], qa)
    write_jsonl(files["strict_qrels"], strict)
    for threshold, rows in multi.items():
        key = f"multi_{threshold:.1f}"
        files[key] = args.output_dir / "qrels" / f"multi_{threshold:.1f}.jsonl"
        write_jsonl(files[key], rows)

    split_audit = []
    for seed in range(1, 11):
        split = create_split(qa, seed)
        validate_split(split, qa)
        split_path = args.output_dir / "splits" / f"seed_{seed:02d}.json"
        write_json(split_path, split)
        files[f"split_{seed:02d}"] = split_path
        split_audit.append(
            {
                "seed": seed,
                "attempt": split["attempt"],
                "documents": {part: len(values) for part, values in split["documents"].items()},
                "qa": {part: len(values) for part, values in split["qids"].items()},
                "all_types_present": {
                    part: len(types) == 8 for part, types in split["question_types"].items()
                },
                "overlap_documents": 0,
            }
        )

    audit = {
        **reconciliation,
        "labels": label_audit,
        "splits": split_audit,
        "privacy": {
            "clinical_text_outputs_are_gitignored": True,
            "aggregate_audit_contains_no_clinical_text": True,
        },
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_dir / "audit.json", audit)

    manifest = {
        "stage": 2,
        "counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "qa": len(qa),
            "strict_qrels": len(strict),
        },
        "files": {
            str(path.relative_to(args.output_dir)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(files.values())
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"reconciliation": audit["reconciliation"], "labels": label_audit, "splits": split_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
