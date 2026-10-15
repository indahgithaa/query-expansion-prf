"""Raw-data reconciliation, deterministic deduplication, and document splits."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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
QUESTION_TYPE_ORDER = tuple(QUESTION_TYPES.values())
HASH_UNICODE_RE = re.compile(r"#U([0-9a-fA-F]{4,6})")


def normalize_filename(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = HASH_UNICODE_RE.sub(lambda match: chr(int(match.group(1), 16)), value)
    return value.replace("\u2060", "")


def make_doc_id(position: int) -> str:
    return f"D{position:04d}"


def make_chunk_id(doc_id: str, page_number: int, chunk_index: int) -> str:
    return f"{doc_id}:P{page_number:04d}:C{chunk_index:03d}"


def make_qid(raw_id: int | str) -> str:
    return f"Q{int(raw_id):04d}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_raw(dataset_path: Path, chunks_dir: Path):
    qa = read_json(dataset_path)
    if not isinstance(qa, list):
        raise TypeError("dataset.json harus berupa list")
    chunk_files = {}
    for path in sorted(chunks_dir.glob("*.json")):
        rows = read_json(path)
        if not isinstance(rows, list):
            raise TypeError(f"{path} harus berupa list")
        chunk_files[path.name] = rows
    if not chunk_files:
        raise ValueError("Tidak ada file chunk")
    return qa, chunk_files


def _unique_normalized_names(names: Iterable[str], source: str) -> dict[str, str]:
    result = {}
    for raw in names:
        normalized = normalize_filename(raw)
        if normalized in result and result[normalized] != raw:
            raise ValueError(f"Collision nama {source}: {result[normalized]!r} dan {raw!r}")
        result[normalized] = raw
    return result


def reconcile_and_process(
    raw_qa: list[dict[str, Any]], chunk_files: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return documents, chunks, deduplicated QA, and privacy-safe audit metadata."""

    qa_names = _unique_normalized_names({row["file"] for row in raw_qa}, "QA")
    chunk_names = _unique_normalized_names(chunk_files, "chunk")
    missing = sorted(set(qa_names) - set(chunk_names))
    extra = sorted(set(chunk_names) - set(qa_names))
    if missing or extra:
        raise ValueError(f"Rekonsiliasi nama gagal: missing={missing}, extra={extra}")

    doc_ids = {name: make_doc_id(index) for index, name in enumerate(sorted(chunk_names), 1)}
    documents = [
        {
            "doc_id": doc_ids[name],
            "normalized_file": name,
            "source_file": chunk_names[name],
        }
        for name in sorted(doc_ids)
    ]

    chunks: list[dict[str, Any]] = []
    chunk_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw_name, rows in sorted(chunk_files.items()):
        name = normalize_filename(raw_name)
        doc_id = doc_ids[name]
        for row in rows:
            page = int(row["page_number"])
            index = int(row["chunk_index"])
            key = (name, page, index)
            if key in chunk_index:
                raise ValueError(f"Kunci chunk duplikat: {key}")
            processed = {
                "chunk_id": make_chunk_id(doc_id, page, index),
                "doc_id": doc_id,
                "page_number": page,
                "chunk_index": index,
                "text": row["text"],
                "source_pdf_name": normalize_filename(row["pdf_name"]),
            }
            chunk_index[key] = processed
            chunks.append(processed)
    chunks.sort(key=lambda row: row["chunk_id"])

    raw_joined = sum(
        (normalize_filename(row["file"]), int(row["page_number"]), int(row["chunk_index"]))
        in chunk_index
        for row in raw_qa
    )
    if raw_joined != len(raw_qa):
        raise ValueError(f"Hanya {raw_joined}/{len(raw_qa)} QA mentah berhasil join")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_qa:
        if row["question"] not in QUESTION_TYPES:
            raise ValueError(f"Template tidak dikenal: {row['question']!r}")
        grouped[(normalize_filename(row["file"]), row["question"])].append(row)

    duplicate_audit = []
    kept = []
    for (filename, question), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["id"]))
        kept.append(ordered[0])
        if len(ordered) > 1:
            locations = {(int(row["page_number"]), int(row["chunk_index"])) for row in ordered}
            duplicate_audit.append(
                {
                    "normalized_file_sha256": sha256_text(filename),
                    "question_type": QUESTION_TYPES[question],
                    "kept_id": int(ordered[0]["id"]),
                    "dropped_ids": [int(row["id"]) for row in ordered[1:]],
                    "group_size": len(ordered),
                    "gold_conflict": len(locations) > 1,
                }
            )

    processed_qa = []
    context_mismatches = []
    for row in sorted(kept, key=lambda item: int(item["id"])):
        filename = normalize_filename(row["file"])
        page = int(row["page_number"])
        index = int(row["chunk_index"])
        chunk = chunk_index[(filename, page, index)]
        context_matches = row["context"] == chunk["text"]
        if not context_matches:
            context_mismatches.append(
                {
                    "qid": make_qid(row["id"]),
                    "chunk_id": chunk["chunk_id"],
                    "source_context_sha256": sha256_text(row["context"]),
                    "chunk_text_sha256": sha256_text(chunk["text"]),
                    "source_context_length": len(row["context"]),
                    "chunk_text_length": len(chunk["text"]),
                }
            )
        processed_qa.append(
            {
                "qid": make_qid(row["id"]),
                "raw_id": int(row["id"]),
                "doc_id": chunk["doc_id"],
                "source_file": filename,
                "report_title": row["report_title"],
                "page_number": page,
                "chunk_index": index,
                "gold_chunk_id": chunk["chunk_id"],
                "question": row["question"],
                "question_type": QUESTION_TYPES[row["question"]],
                "answer_texts": list(row["answer"]["text"]),
                "medical_diagnosis": row["medical_diagnosis"],
                "anthropometric_assessment": row["anthropometric_assessment"],
                "source_context_matches_chunk": context_matches,
            }
        )

    audit = {
        "reconciliation": {
            "raw_qa_rows": len(raw_qa),
            "raw_qa_documents": len(qa_names),
            "raw_chunks": len(chunks),
            "raw_qa_joined": raw_joined,
            "duplicate_groups": len(duplicate_audit),
            "duplicate_rows_removed": sum(len(item["dropped_ids"]) for item in duplicate_audit),
            "duplicate_groups_with_gold_conflict": sum(item["gold_conflict"] for item in duplicate_audit),
            "qa_after_deduplication": len(kept),
            "qa_after_join": len(processed_qa),
            "final_qa_rows": len(processed_qa),
            "context_mismatches_final": len(context_mismatches),
            "filename_changes_qa": sum(normalize_filename(name) != name for name in {row["file"] for row in raw_qa}),
            "filename_changes_chunks": sum(normalize_filename(name) != name for name in chunk_files),
        },
        "deduplication_policy": "keep_smallest_numeric_id_per_normalized_file_and_question",
        "duplicate_groups": duplicate_audit,
        "context_mismatches": context_mismatches,
    }
    return documents, chunks, processed_qa, audit


def create_split(qa: list[dict[str, Any]], seed: int, max_attempts: int = 10000) -> dict[str, Any]:
    """Find a deterministic 72/24/24 document split containing every type."""

    doc_types: dict[str, set[str]] = defaultdict(set)
    for row in qa:
        doc_types[row["doc_id"]].add(row["question_type"])
    docs = sorted(doc_types)
    if len(docs) != 120:
        raise ValueError(f"Diharapkan 120 dokumen, ditemukan {len(docs)}")
    required = set(QUESTION_TYPE_ORDER)

    for attempt in range(max_attempts):
        shuffled = docs.copy()
        random.Random(seed * 1_000_003 + attempt).shuffle(shuffled)
        parts = {
            "collection": sorted(shuffled[:72]),
            "dev": sorted(shuffled[72:96]),
            "test": sorted(shuffled[96:]),
        }
        present = {
            part: set().union(*(doc_types[doc_id] for doc_id in part_docs))
            for part, part_docs in parts.items()
        }
        if all(present[part] == required for part in parts):
            qids = {
                part: sorted(row["qid"] for row in qa if row["doc_id"] in set(part_docs))
                for part, part_docs in parts.items()
            }
            return {
                "seed": seed,
                "attempt": attempt,
                "documents": parts,
                "qids": qids,
                "question_types": {part: sorted(present[part]) for part in parts},
            }
    raise RuntimeError(f"Tidak menemukan split valid untuk seed {seed} setelah {max_attempts} percobaan")


def validate_split(split: dict[str, Any], qa: list[dict[str, Any]]) -> None:
    parts = split["documents"]
    sets = {name: set(values) for name, values in parts.items()}
    if any(sets[a] & sets[b] for a, b in (("collection", "dev"), ("collection", "test"), ("dev", "test"))):
        raise AssertionError("Tumpang tindih dokumen antarbagian")
    if len(set().union(*sets.values())) != 120:
        raise AssertionError("Cakupan split bukan 120 dokumen")
    qid_to_row = {row["qid"]: row for row in qa}
    expected_types = set(QUESTION_TYPE_ORDER)
    for part, qids in split["qids"].items():
        if {qid_to_row[qid]["question_type"] for qid in qids} != expected_types:
            raise AssertionError(f"Tipe tidak lengkap pada {part}")
