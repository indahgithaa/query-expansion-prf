"""Audit the raw QA/chunk sources and create aggregate Stage 1 evidence."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


QUESTION_TYPES = {
    "Apa metode skrining status gizi yang direkomendasikan?": "skrining",
    "Bagaimana status gizi pada pasien ini?": "status_gizi",
    "Bagaimana data biokimia/parameter lab pada pasien (yg abnormal)?": "biokimia",
    "Bagaimana menyusun diagnosis gizi dengan format PES pada pasien ini?": "pes",
    "Apa tujuan diet pada pasien ini?": "tujuan_diet",
    "Bagaimana prinsip diet yang tepat sesuai dengan kondisi pasien ini?": "prinsip_diet",
    "Bagaimana syarat diet yang harus dipenuhi dalam pengaturan diet pasien ini?": "syarat_diet",
    "Apa pedoman diet yang digunakan dalam perencanaan diet pasien ini?": "pedoman",
}
TYPE_ORDER = tuple(QUESTION_TYPES.values())
QA_FIELDS = {
    "id", "file", "page_number", "chunk_index", "question", "context", "answer",
    "medical_diagnosis", "anthropometric_assessment", "report_title",
}
CHUNK_FIELDS = {"pdf_name", "page_number", "chunk_index", "text"}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
HASH_UNICODE_RE = re.compile(r"#U([0-9a-fA-F]{4,6})")


def normalize_filename(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = HASH_UNICODE_RE.sub(lambda match: chr(int(match.group(1), 16)), value)
    return value.replace("\u2060", "")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def describe(values: Iterable[int]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    return {
        "n": int(array.size),
        "min": int(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q3": float(np.quantile(array, 0.75)),
        "max": int(array.max()),
    }


def missing_counts(rows: list[dict[str, Any]], fields: set[str]) -> dict[str, int]:
    result = {}
    for field in sorted(fields):
        result[field] = sum(
            field not in row
            or row[field] is None
            or (isinstance(row[field], (str, list, dict)) and not row[field])
            for row in rows
        )
    return result


def load_sources(dataset: Path, chunks_dir: Path):
    qa = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(qa, list):
        raise TypeError("dataset.json harus berupa list JSON")
    chunk_files = {}
    for path in sorted(chunks_dir.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError(f"{path} harus berupa list JSON")
        chunk_files[path.name] = rows
    if not chunk_files:
        raise FileNotFoundError(f"Tidak ada file chunk di {chunks_dir}")
    return qa, chunk_files


def audit(qa: list[dict[str, Any]], chunk_files: dict[str, list[dict[str, Any]]]):
    chunks = [row for rows in chunk_files.values() for row in rows]
    question_counts = Counter(row["question"] for row in qa)
    unknown = sorted(set(question_counts) - set(QUESTION_TYPES))
    if unknown:
        raise ValueError(f"Template belum dipetakan: {unknown}")

    type_counts = Counter(QUESTION_TYPES[row["question"]] for row in qa)
    type_docs = {
        kind: len({row["file"] for row in qa if QUESTION_TYPES[row["question"]] == kind})
        for kind in TYPE_ORDER
    }
    duplicate_counts = Counter((row["file"], row["question"]) for row in qa)

    raw_chunk_names = set(chunk_files)
    norm_chunk_names: dict[str, list[str]] = {}
    for name in chunk_files:
        norm_chunk_names.setdefault(normalize_filename(name), []).append(name)
    qa_names = {row["file"] for row in qa}

    chunk_by_raw_key = {}
    chunk_by_norm_key = {}
    duplicate_chunk_keys = 0
    for filename, rows in chunk_files.items():
        for row in rows:
            key = (filename, int(row["page_number"]), int(row["chunk_index"]))
            duplicate_chunk_keys += key in chunk_by_raw_key
            chunk_by_raw_key[key] = row
            norm_key = (
                normalize_filename(filename), int(row["page_number"]), int(row["chunk_index"])
            )
            if norm_key in chunk_by_norm_key:
                raise ValueError(f"Collision kunci chunk setelah normalisasi: {norm_key}")
            chunk_by_norm_key[norm_key] = row

    raw_join = normalized_join = context_match = 0
    answer_texts: list[str] = []
    exact_offsets = found_verbatim = invalid_offset_qa = length_mismatches = 0
    for row in qa:
        raw_key = (row["file"], int(row["page_number"]), int(row["chunk_index"]))
        norm_key = (
            normalize_filename(row["file"]), int(row["page_number"]), int(row["chunk_index"])
        )
        raw_join += raw_key in chunk_by_raw_key
        matched = chunk_by_norm_key.get(norm_key)
        normalized_join += matched is not None
        context_match += matched is not None and matched["text"] == row["context"]

        texts = list(row["answer"].get("text", []))
        starts = list(row["answer"].get("answer_start", []))
        length_mismatches += len(texts) != len(starts)
        any_invalid = False
        for index, text in enumerate(texts):
            answer_texts.append(text)
            found_verbatim += text in row["context"]
            start = starts[index] if index < len(starts) else None
            valid = (
                isinstance(start, int)
                and start >= 0
                and row["context"][start : start + len(text)] == text
            )
            exact_offsets += valid
            any_invalid = any_invalid or not valid
        invalid_offset_qa += any_invalid

    qa_per_doc = Counter(row["file"] for row in qa)
    chunks_per_doc = {name: len(rows) for name, rows in chunk_files.items()}
    filename_anomalies = lambda names: {
        "hash_unicode": sum(bool(HASH_UNICODE_RE.search(name)) for name in names),
        "word_joiner_u2060": sum("\u2060" in name for name in names),
        "changed_by_nfkc": sum(unicodedata.normalize("NFKC", name) != name for name in names),
        "changed_by_full_normalization": sum(normalize_filename(name) != name for name in names),
    }

    summary = {
        "source": {
            "dataset": "data/dataset/dataset.json",
            "chunks_directory": "data/chunks",
            "stage": 1,
            "raw_read_only": True,
        },
        "counts": {
            "qa_rows": len(qa),
            "unique_qa_ids": len({str(row["id"]) for row in qa}),
            "qa_documents": len(qa_names),
            "chunk_files": len(chunk_files),
            "chunks": len(chunks),
            "chunk_pdf_names": len({row["pdf_name"] for row in chunks}),
            "question_templates": len(question_counts),
        },
        "schema": {
            "qa_fields": sorted(set().union(*(row.keys() for row in qa))),
            "chunk_fields": sorted(set().union(*(row.keys() for row in chunks))),
            "qa_unexpected_field_sets": sum(set(row) != QA_FIELDS for row in qa),
            "chunk_unexpected_field_sets": sum(set(row) != CHUNK_FIELDS for row in chunks),
            "qa_missing_or_empty": missing_counts(qa, QA_FIELDS),
            "chunk_missing_or_empty": missing_counts(chunks, CHUNK_FIELDS),
        },
        "questions": {
            "template_counts": dict(sorted(question_counts.items())),
            "type_counts": {kind: type_counts[kind] for kind in TYPE_ORDER},
            "type_document_counts": type_docs,
            "unknown_templates": unknown,
        },
        "duplicates": {
            "key": ["file", "question"],
            "groups": sum(count > 1 for count in duplicate_counts.values()),
            "extra_rows": sum(count - 1 for count in duplicate_counts.values()),
            "max_group_size": max(duplicate_counts.values()),
            "expected_rows_after_deduplication": len(duplicate_counts),
        },
        "filenames": {
            "raw_exact_matches": sum(name in raw_chunk_names for name in qa_names),
            "normalized_matches": sum(normalize_filename(name) in norm_chunk_names for name in qa_names),
            "qa_anomalies": filename_anomalies(qa_names),
            "chunk_anomalies": filename_anomalies(raw_chunk_names),
            "normalization_collisions": sum(len(names) > 1 for names in norm_chunk_names.values()),
        },
        "joins": {
            "duplicate_chunk_keys": duplicate_chunk_keys,
            "raw_key_matches": raw_join,
            "normalized_key_matches": normalized_join,
            "context_exact_matches": context_match,
            "context_mismatches": normalized_join - context_match,
        },
        "answers": {
            "spans": len(answer_texts),
            "exact_valid_offsets": exact_offsets,
            "invalid_offsets": len(answer_texts) - exact_offsets,
            "qa_with_invalid_offset": invalid_offset_qa,
            "verbatim_in_context": found_verbatim,
            "not_verbatim_in_context": len(answer_texts) - found_verbatim,
            "array_length_mismatches": length_mismatches,
        },
        "lengths": {
            "question_tokens": describe(len(tokens(row["question"])) for row in qa),
            "answer_tokens": describe(len(tokens(text)) for text in answer_texts),
            "answer_characters": describe(len(text) for text in answer_texts),
            "context_tokens": describe(len(tokens(row["context"])) for row in qa),
            "context_characters": describe(len(row["context"]) for row in qa),
            "chunk_tokens": describe(len(tokens(row["text"])) for row in chunks),
            "chunk_characters": describe(len(row["text"]) for row in chunks),
        },
        "per_document": {
            "qa": describe(qa_per_doc.values()),
            "chunks": describe(chunks_per_doc.values()),
        },
    }
    plot_data = {
        "answer_tokens": [len(tokens(text)) for text in answer_texts],
        "qa_per_doc": [qa_per_doc[name] for name in sorted(qa_names)],
        "chunks_per_doc": [chunks_per_doc[name] for name in sorted(qa_names)],
    }
    return summary, plot_data


def plot(summary: dict[str, Any], values: dict[str, list[int]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    counts = [summary["questions"]["type_counts"][kind] for kind in TYPE_ORDER]
    bars = ax.barh([kind.replace("_", " ") for kind in TYPE_ORDER], counts, color="#35618f")
    ax.invert_yaxis(); ax.set(xlabel="Jumlah QA", ylabel="Tipe pertanyaan", title="Distribusi tipe pertanyaan mentah")
    ax.bar_label(bars, padding=3, fontsize=8); fig.tight_layout()
    fig.savefig(output / "01_question_type_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    answer_lengths = np.asarray(values["answer_tokens"])
    clip = int(np.quantile(answer_lengths, 0.99))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(np.clip(answer_lengths, 0, clip), bins=40, color="#c66b3d", edgecolor="white")
    ax.axvline(np.median(answer_lengths), color="#222", linestyle="--", linewidth=1.2)
    ax.set(xlabel=f"Panjang jawaban (token; >{clip} digabung)", ylabel="Jumlah jawaban", title="Distribusi panjang jawaban")
    fig.tight_layout(); fig.savefig(output / "02_answer_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(values["chunks_per_doc"], values["qa_per_doc"], alpha=0.72, color="#3d8f78", edgecolors="white", linewidths=0.4)
    ax.set(xlabel="Jumlah chunk per dokumen", ylabel="Jumlah QA per dokumen", title="Cakupan QA dan chunk per dokumen")
    fig.tight_layout(); fig.savefig(output / "03_document_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset/dataset.json"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--report-summary", type=Path, default=Path("reports/01_eda_summary.json"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    qa, chunk_files = load_sources(args.dataset, args.chunks_dir)
    summary, plot_values = audit(qa, chunk_files)
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(payload, encoding="utf-8")
    args.report_summary.parent.mkdir(parents=True, exist_ok=True)
    args.report_summary.write_text(payload, encoding="utf-8")
    plot(summary, plot_values, args.figures_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
