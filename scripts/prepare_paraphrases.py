"""Create and validate 96 Indonesian paraphrases and scenario manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import QUESTION_TYPE_ORDER, read_json, read_jsonl, write_json, write_jsonl
from src.scenarios import SCENARIOS, scenario_collection, scenario_query
from src.text import tokenize


PARAPHRASES = {
    "skrining": {
        "dev": [
            "Alat apa yang sebaiknya dipakai untuk menyaring risiko gizi pasien?",
            "Metode penapisan gizi mana yang dianjurkan untuk kasus ini?",
            "Bagaimana cara melakukan skrining risiko malnutrisi pada pasien tersebut?",
            "Instrumen skrining gizi apa yang sesuai bagi pasien ini?",
            "Penilaian awal risiko gizi pasien menggunakan metode apa?",
            "Apa pilihan alat deteksi dini masalah gizi untuk pasien tersebut?",
        ],
        "test": [
            "Dengan instrumen apa risiko kekurangan gizi pasien perlu ditapis?",
            "Skrining gizi yang cocok untuk kondisi pasien ini memakai alat apa?",
            "Apa metode yang digunakan untuk penapisan status gizi awal pasien?",
            "Bagaimana risiko malnutrisi pasien ini sebaiknya diskrining?",
            "Alat penilaian cepat gizi mana yang direkomendasikan dalam kasus ini?",
            "Untuk mendeteksi risiko gizi, instrumen apa yang perlu digunakan?",
        ],
    },
    "status_gizi": {
        "dev": [
            "Seperti apa keadaan gizi pasien berdasarkan hasil pengkajian?",
            "Pasien ini termasuk kategori status nutrisi yang mana?",
            "Bagaimana klasifikasi kondisi gizi pasien tersebut?",
            "Apa kesimpulan penilaian status nutrisi pada kasus ini?",
            "Apakah kondisi gizi pasien tergolong normal atau bermasalah?",
            "Bagaimana gambaran status gizi pasien dari data antropometri?",
        ],
        "test": [
            "Kategori gizi apa yang ditetapkan untuk pasien ini?",
            "Dari hasil pengukuran, bagaimana keadaan nutrisi pasien?",
            "Apa interpretasi status gizi pada pasien tersebut?",
            "Bagaimana hasil klasifikasi gizi pasien dalam laporan ini?",
            "Pasien tersebut dinilai memiliki kondisi gizi seperti apa?",
            "Apa simpulan mengenai keadaan gizi pasien berdasarkan asesmen?",
        ],
    },
    "biokimia": {
        "dev": [
            "Bagaimana hasil pemeriksaan laboratorium pasien, terutama nilai yang tidak normal?",
            "Parameter biokimia apa yang menyimpang pada pasien ini?",
            "Apa saja temuan hasil cek darah atau laboratorium pada kasus tersebut?",
            "Nilai lab pasien mana yang berada di luar batas normal?",
            "Bagaimana gambaran data biokimia pasien berdasarkan pemeriksaan?",
            "Adakah hasil laboratorium abnormal yang perlu diperhatikan pada pasien?",
        ],
        "test": [
            "Apakah ada hasil pemeriksaan darah pasien yang di luar batas normal?",
            "Temuan biokimia apa yang tercatat dalam laporan pasien ini?",
            "Bagaimana interpretasi nilai-nilai laboratorium pada kasus tersebut?",
            "Parameter pemeriksaan klinis mana yang menunjukkan kelainan biokimia?",
            "Apa hasil tes laboratorium yang bermasalah pada pasien ini?",
            "Nilai biokimia pasien seperti apa yang dinyatakan abnormal?",
        ],
    },
    "pes": {
        "dev": [
            "Apa diagnosis gizi pasien bila ditulis dalam susunan PES?",
            "Bagaimana rumusan masalah, etiologi, dan tanda gejala gizi pasien ini?",
            "Susunlah diagnosis nutrisi pasien dengan format Problem Etiology Signs Symptoms.",
            "Pernyataan PES apa yang sesuai untuk kondisi pasien tersebut?",
            "Bagaimana diagnosis gizi terstandar pasien ini dirumuskan?",
            "Apa rumusan problem, penyebab, serta bukti masalah gizi pasien?",
        ],
        "test": [
            "Bagaimana penulisan diagnosis gizi lengkap dalam bentuk PES untuk kasus ini?",
            "Apa masalah gizi, penyebab, dan tanda gejala yang dialami pasien?",
            "Rumusan PES manakah yang menggambarkan kondisi nutrisi pasien ini?",
            "Tolong nyatakan diagnosis nutrisi pasien memakai struktur PES.",
            "Bagaimana problem gizi dikaitkan dengan etiologi dan gejalanya pada pasien?",
            "Apa diagnosis gizi terstruktur yang tepat bagi pasien tersebut?",
        ],
    },
    "tujuan_diet": {
        "dev": [
            "Diet pasien ini ditujukan untuk apa?",
            "Apa target pemberian terapi gizi pada pasien tersebut?",
            "Hasil apa yang ingin dicapai melalui pengaturan makan pasien?",
            "Apa maksud utama intervensi diet dalam kasus ini?",
            "Untuk mencapai sasaran apa diet pasien dirancang?",
            "Apa tujuan penatalaksanaan makanan bagi pasien tersebut?",
        ],
        "test": [
            "Apa saja sasaran dari terapi diet pasien ini?",
            "Mengapa pengaturan diet diberikan kepada pasien tersebut?",
            "Apa target klinis yang hendak dicapai lewat intervensi gizi?",
            "Pemberian makanan pasien diarahkan untuk mencapai tujuan apa?",
            "Apa hasil yang diharapkan dari terapi nutrisi pada kasus ini?",
            "Untuk apa rencana diet tersebut diterapkan pada pasien?",
        ],
    },
    "prinsip_diet": {
        "dev": [
            "Apa prinsip utama pengaturan makanan yang sesuai dengan kondisi pasien?",
            "Bagaimana kaidah dasar terapi diet untuk kasus ini?",
            "Prinsip pemberian nutrisi apa yang perlu diterapkan pada pasien?",
            "Apa dasar pengaturan diet yang tepat bagi pasien tersebut?",
            "Bagaimana pendekatan umum diet disesuaikan dengan penyakit pasien?",
            "Kaidah diet apa yang menjadi acuan dalam penanganan pasien ini?",
        ],
        "test": [
            "Apa prinsip terapi gizi yang dianjurkan untuk pasien tersebut?",
            "Bagaimana dasar penyusunan diet sesuai keadaan klinis pasien?",
            "Pedoman umum pemberian makanan apa yang berlaku pada kasus ini?",
            "Prinsip nutrisi mana yang harus diikuti dalam diet pasien?",
            "Bagaimana pola dasar intervensi diet bagi kondisi pasien tersebut?",
            "Apa kaidah utama diet yang sesuai untuk penyakit pasien ini?",
        ],
    },
    "syarat_diet": {
        "dev": [
            "Ketentuan apa saja yang perlu dipenuhi dalam diet pasien ini?",
            "Bagaimana persyaratan zat gizi untuk pengaturan makan pasien tersebut?",
            "Apa batasan dan kebutuhan yang harus ada dalam terapi diet pasien?",
            "Syarat pemberian makanan apa yang sesuai bagi kondisi pasien ini?",
            "Bagaimana ketentuan energi, protein, dan zat gizi lain dalam diet pasien?",
            "Apa persyaratan khusus rencana makan untuk kasus tersebut?",
        ],
        "test": [
            "Diet pasien ini harus memenuhi ketentuan gizi apa saja?",
            "Apa saja syarat yang perlu diperhatikan saat menyusun makanan pasien?",
            "Bagaimana kebutuhan dan pembatasan nutrisi dalam diet kasus ini?",
            "Ketentuan terapi makanan apa yang diterapkan kepada pasien tersebut?",
            "Apa persyaratan komposisi diet sesuai kondisi klinis pasien?",
            "Bagaimana syarat energi dan zat gizi untuk pasien dalam laporan ini?",
        ],
    },
    "pedoman": {
        "dev": [
            "Acuan diet apa yang dipakai untuk merencanakan makanan pasien ini?",
            "Pedoman gizi mana yang menjadi dasar penyusunan diet pasien?",
            "Rencana makan pasien tersebut mengikuti panduan diet apa?",
            "Apa standar diet yang digunakan dalam perencanaan terapi gizi kasus ini?",
            "Panduan pemberian makanan apa yang dirujuk untuk pasien tersebut?",
            "Perencanaan diet pasien mengacu pada pedoman yang mana?",
        ],
        "test": [
            "Apa acuan yang digunakan ketika menyusun rencana diet pasien?",
            "Panduan diet mana yang diterapkan dalam penatalaksanaan gizi kasus ini?",
            "Standar perencanaan makanan apa yang dipakai untuk pasien tersebut?",
            "Terapi gizi pasien disusun berdasarkan pedoman apa?",
            "Apa rujukan diet yang mendasari pengaturan makan pasien ini?",
            "Pedoman nutrisi apa yang dipilih untuk membuat rencana makan pasien?",
        ],
    },
}

INTENT_TERMS = {
    "skrining": {"skrining", "penapisan", "risiko", "alat", "deteksi", "penilaian"},
    "status_gizi": {"status", "gizi", "nutrisi", "keadaan"},
    "biokimia": {"biokimia", "laboratorium", "lab", "darah"},
    "pes": {"pes", "diagnosis", "problem", "etiologi", "masalah", "penyebab", "gejala"},
    "tujuan_diet": {"tujuan", "target", "sasaran", "dicapai", "ditujukan", "diet", "hasil", "diharapkan", "terapi"},
    "prinsip_diet": {"prinsip", "kaidah", "dasar", "pedoman", "pendekatan", "pola"},
    "syarat_diet": {"syarat", "ketentuan", "persyaratan", "kebutuhan"},
    "pedoman": {"pedoman", "panduan", "acuan", "standar", "rujukan"},
}


def validate(paraphrases: dict, templates: dict[str, str]) -> dict:
    all_dev, all_test = set(), set()
    details = {}
    for question_type in QUESTION_TYPE_ORDER:
        groups = paraphrases[question_type]
        assert len(groups["dev"]) == len(groups["test"]) == 6
        dev = {text.casefold().strip() for text in groups["dev"]}
        test = {text.casefold().strip() for text in groups["test"]}
        assert len(dev) == len(test) == 6 and not dev & test
        original = templates[question_type].casefold().strip()
        assert original not in dev | test
        for text in dev | test:
            tokens = set(tokenize(text))
            assert tokens & INTENT_TERMS[question_type], (question_type, text)
        all_dev |= dev
        all_test |= test
        details[question_type] = {"dev": 6, "test": 6, "overlap": 0, "intent_check": "PASS"}
    assert not all_dev & all_test
    assert len(all_dev) == len(all_test) == 48
    return {"status": "PASS", "total": 96, "dev": 48, "test": 48, "details": details}


def report_text(paraphrases: dict, audit: dict) -> str:
    lines = [
        "# Set Parafrasa dan Skenario", "", "Tanggal: 15 Oktober 2026", "",
        "## Ringkasan validasi", "",
        f"- Total: {audit['total']} parafrasa ({audit['dev']} dev dan {audit['test']} test).",
        "- Tidak ada parafrasa identik dengan template asli.",
        "- Tidak ada duplikasi atau tumpang tindih dev–test.",
        "- Pemeriksaan kata penanda maksud lulus untuk seluruh parafrasa.",
        "- Pada NEW dan PNEW, seluruh QA dengan tipe target dikeluarkan dari koleksi pertanyaan.", "",
    ]
    for question_type in QUESTION_TYPE_ORDER:
        lines.extend([f"## {question_type}", "", "### Dev", ""])
        lines.extend(f"{index}. {text}" for index, text in enumerate(paraphrases[question_type]["dev"], 1))
        lines.extend(["", "### Test", ""])
        lines.extend(f"{index}. {text}" for index, text in enumerate(paraphrases[question_type]["test"], 1))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results/scenarios"))
    args = parser.parse_args()
    qa = read_jsonl(args.processed / "qa.jsonl")
    templates = {row["question_type"]: row["question"] for row in qa}
    audit = validate(PARAPHRASES, templates)
    write_json(args.processed / "paraphrases.json", PARAPHRASES)
    scenario_audits = []
    for seed in range(1, 11):
        split = read_json(args.processed / "splits" / f"seed_{seed:02d}.json")
        qid_to_part = {qid: part for part in ("collection", "dev", "test") for qid in split["qids"][part]}
        collection = [row for row in qa if qid_to_part[row["qid"]] == "collection"]
        rows = []
        for row in qa:
            part = qid_to_part[row["qid"]]
            if part not in {"dev", "test"}:
                continue
            tagged = {**row, "partition": part}
            for scenario in SCENARIOS:
                source = scenario_collection(collection, row["question_type"], scenario)
                rows.append({
                    "seed": seed, "partition": part, "scenario": scenario,
                    "qid": row["qid"], "doc_id": row["doc_id"],
                    "question_type": row["question_type"],
                    "query": scenario_query(tagged, scenario, seed, PARAPHRASES),
                    "gold_chunk_id": row["gold_chunk_id"],
                    "feedback_question_count": len(source),
                    "excluded_question_type": row["question_type"] if scenario in {"NEW", "PNEW"} else None,
                })
                if scenario in {"NEW", "PNEW"}:
                    assert not any(item["question_type"] == row["question_type"] for item in source)
        write_jsonl(args.processed / "scenarios" / f"seed_{seed:02d}.jsonl", rows)
        scenario_audits.append({"seed": seed, "rows": len(rows), "new_pnew_checks": sum(r["scenario"] in {"NEW", "PNEW"} for r in rows)})
    result = {**audit, "seeds": scenario_audits}
    write_json(args.results / "audit.json", result)
    Path("reports/04_paraphrases.md").write_text(report_text(PARAPHRASES, result) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
