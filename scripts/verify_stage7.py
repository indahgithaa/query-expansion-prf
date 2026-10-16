"""Verify completeness and referential integrity of all Stage 7 outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage7 import METHODS, file_sha256
from src.data import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    chunks = read_jsonl(args.processed / "chunks.jsonl")
    chunk_doc = {row["chunk_id"]: row["doc_id"] for row in chunks}
    chunks_per_doc = Counter(row["doc_id"] for row in chunks)
    total_queries = 0
    total_rankings = 0
    short_documents = set()
    seeds = []

    for seed in range(1, 11):
        param_path = args.results / "params" / f"seed_{seed:02d}.yaml"
        parameters = yaml.safe_load(param_path.read_text(encoding="utf-8"))
        if parameters["seed"] != seed:
            raise AssertionError(f"Seed parameter tidak cocok: {param_path}")
        if (
            parameters["selection_partition"] != "dev"
            or parameters["evaluation_partition"] != "test"
            or parameters["test_used_for_tuning"] is not False
        ):
            raise AssertionError(f"Audit partisi gagal untuk seed {seed}")
        outputs = parameters["outputs"]
        ranking_path = Path(outputs["ranking_file"])
        query_path = Path(outputs["query_file"])
        if file_sha256(ranking_path) != outputs["ranking_sha256"]:
            raise AssertionError(f"Hash ranking seed {seed} berubah")
        if file_sha256(query_path) != outputs["query_sha256"]:
            raise AssertionError(f"Hash query seed {seed} berubah")

        scenarios = [
            row for row in read_jsonl(args.processed / "scenarios" / f"seed_{seed:02d}.jsonl")
            if row["partition"] == "test"
        ]
        expected = {
            (row["qid"], row["scenario"]): row
            for row in scenarios
        }
        query_rows = list(csv.DictReader(gzip.open(query_path, "rt", encoding="utf-8")))
        query_lookup = {(row["qid"], row["scenario"]): row for row in query_rows}
        if len(query_rows) != len(expected) or set(query_lookup) != set(expected):
            raise AssertionError(f"Cakupan query seed {seed} tidak cocok")

        seen: dict[tuple[str, str], set[str]] = defaultdict(set)
        rows = 0
        with gzip.open(ranking_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["qid"], row["scenario"])
                if row["seed"] != seed or row["partition"] != "test" or key not in expected:
                    raise AssertionError(f"Identitas ranking tidak valid pada seed {seed}")
                source = expected[key]
                if (
                    row["doc_id"] != source["doc_id"]
                    or row["gold_chunk_id"] != source["gold_chunk_id"]
                    or row["question_type"] != source["question_type"]
                ):
                    raise AssertionError(f"Referensi ranking tidak cocok: seed {seed}, {key}")
                method = row["method"]
                if method not in METHODS or method in seen[key]:
                    raise AssertionError(f"Metode hilang/duplikat: seed {seed}, {key}, {method}")
                seen[key].add(method)
                ranking = row["ranking"]
                expected_length = min(50, chunks_per_doc[row["doc_id"]])
                if len(ranking) != expected_length or len(ranking) != len(set(ranking)):
                    raise AssertionError(f"Panjang/duplikasi ranking gagal: seed {seed}, {key}, {method}")
                if any(chunk_doc.get(chunk_id) != row["doc_id"] for chunk_id in ranking):
                    raise AssertionError(f"Ranking keluar dari dokumen pasien: seed {seed}, {key}")
                rank = int(query_lookup[key][f"rank__{method}"])
                if not 1 <= rank <= chunks_per_doc[row["doc_id"]]:
                    raise AssertionError(f"Rank di luar rentang: seed {seed}, {key}, {method}")
                if rank <= expected_length:
                    if ranking[rank - 1] != row["gold_chunk_id"]:
                        raise AssertionError(f"Posisi gold tidak cocok: seed {seed}, {key}, {method}")
                elif row["gold_chunk_id"] in ranking:
                    raise AssertionError(f"Gold top-50 tidak cocok dengan rank: seed {seed}, {key}, {method}")
                rows += 1

        if set(seen) != set(expected) or any(methods != set(METHODS) for methods in seen.values()):
            raise AssertionError(f"Cakupan metode seed {seed} tidak lengkap")
        if rows != len(expected) * len(METHODS) or rows != int(outputs["ranking_rows"]):
            raise AssertionError(f"Jumlah ranking seed {seed} tidak cocok")
        if int(outputs["test_queries"]) != len(expected):
            raise AssertionError(f"Jumlah query parameter seed {seed} tidak cocok")
        short_documents.update(
            row["doc_id"] for row in scenarios if chunks_per_doc[row["doc_id"]] < 50
        )
        total_queries += len(expected)
        total_rankings += rows
        seeds.append(seed)

    audit = json.loads((args.results / "stage7" / "audit.json").read_text(encoding="utf-8"))
    if (
        audit["status"] != "PASS"
        or audit["seeds"] != seeds
        or audit["completed_seeds"] != 10
        or audit["test_used_for_tuning"] is not False
        or audit["total_test_queries"] != total_queries
        or audit["total_ranking_rows"] != total_rankings
    ):
        raise AssertionError("Audit utama Tahap 7 tidak cocok dengan artefak")
    result = {
        "status": "PASS",
        "seeds": seeds,
        "methods": len(METHODS),
        "test_queries": total_queries,
        "ranking_rows": total_rankings,
        "short_documents": len(short_documents),
        "test_used_for_tuning": False,
        "hashes_verified": 20,
    }
    write_json(args.results / "stage7" / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
