"""Benchmark and cache normalized multilingual E5 embeddings for Stage 3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import read_jsonl, write_json
from src.dense import E5Encoder, validate_embeddings


def resolved_revision(model_id: str) -> str:
    try:
        from huggingface_hub import model_info

        return str(model_info(model_id).sha)
    except Exception as exc:  # cache-only execution remains reproducible through model_id and file hashes
        return f"unresolved:{type(exc).__name__}"


def encode_deduplicated(encoder: E5Encoder, texts: list[str], *, kind: str, progress: bool):
    unique = list(dict.fromkeys(texts))
    vectors = encoder.encode(unique, kind=kind, progress=progress)
    position = {text: index for index, text in enumerate(unique)}
    expanded = np.asarray([vectors[position[text]] for text in texts], dtype=np.float32)
    return validate_embeddings(expanded, rows=len(texts)), len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--results", type=Path, default=Path("results/embeddings"))
    parser.add_argument("--cache-folder", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--fallback-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--benchmark-size", type=int, default=64)
    parser.add_argument("--max-estimated-seconds", type=float, default=7200.0)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    qa = read_jsonl(args.processed / "qa.jsonl")
    chunks = read_jsonl(args.processed / "chunks.jsonl")
    model_id = args.model
    fallback_reason = None
    started_load = time.perf_counter()
    try:
        encoder = E5Encoder(model_id, cache_folder=args.cache_folder, batch_size=args.batch_size)
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
        model_id = args.fallback_model
        encoder = E5Encoder(model_id, cache_folder=args.cache_folder, batch_size=args.batch_size)
    load_seconds = time.perf_counter() - started_load

    sample_count = min(args.benchmark_size, len(chunks))
    step = max(1, len(chunks) // sample_count)
    sample = [chunks[index]["text"] for index in range(0, len(chunks), step)][:sample_count]
    encoder.encode(sample[: min(4, sample_count)], kind="passage")
    benchmark_start = time.perf_counter()
    encoder.encode(sample, kind="passage")
    benchmark_seconds = time.perf_counter() - benchmark_start
    unique_question_count = len({row["question"] for row in qa})
    unique_answer_count = len({" ".join(row["answer_texts"]) for row in qa})
    items_total = len(chunks) + unique_question_count + unique_answer_count
    estimated_seconds = benchmark_seconds / sample_count * items_total * 1.15
    revision = resolved_revision(model_id)
    benchmark = {
        "model_id": model_id,
        "revision": revision,
        "fallback_reason": fallback_reason,
        "device": "cpu",
        "load_seconds": load_seconds,
        "sample_items": sample_count,
        "sample_seconds": benchmark_seconds,
        "items_per_second": sample_count / benchmark_seconds,
        "estimated_total_seconds_with_15_percent_margin": estimated_seconds,
        "estimated_unique_items": {
            "chunks": len(chunks),
            "questions": unique_question_count,
            "answers": unique_answer_count,
            "total": items_total,
        },
        "cpu_limit_seconds": args.max_estimated_seconds,
        "cpu_allowed": estimated_seconds <= args.max_estimated_seconds,
        "dimension": encoder.dimension,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    write_json(args.results / "benchmark.json", benchmark)
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))
    if args.benchmark_only:
        return
    if estimated_seconds > args.max_estimated_seconds:
        raise RuntimeError("Estimasi embedding CPU melebihi dua jam; gunakan Kaggle")

    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    chunk_vectors = encoder.encode([row["text"] for row in chunks], kind="passage", progress=True)
    question_vectors, unique_questions = encode_deduplicated(
        encoder, [row["question"] for row in qa], kind="query", progress=False
    )
    answer_vectors, unique_answers = encode_deduplicated(
        encoder, [" ".join(row["answer_texts"]) for row in qa], kind="passage", progress=True
    )
    elapsed = time.perf_counter() - start
    np.save(args.output / "chunks.npy", chunk_vectors)
    np.save(args.output / "questions.npy", question_vectors)
    np.save(args.output / "answers.npy", answer_vectors)
    write_json(args.output / "ids.json", {
        "chunk_ids": [row["chunk_id"] for row in chunks],
        "qids": [row["qid"] for row in qa],
    })
    audit = {
        **benchmark,
        "actual_encoding_seconds": elapsed,
        "rows": {"chunks": len(chunks), "questions": len(qa), "answers": len(qa)},
        "unique_texts": {"questions": unique_questions, "answers": unique_answers},
        "normalized": True,
        "files": {
            name: {"shape": list(values.shape), "dtype": str(values.dtype)}
            for name, values in (
                ("chunks.npy", chunk_vectors), ("questions.npy", question_vectors), ("answers.npy", answer_vectors)
            )
        },
        "paraphrase_embeddings_pending_stage_4": True,
    }
    write_json(args.output / "manifest.json", audit)
    write_json(args.results / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
