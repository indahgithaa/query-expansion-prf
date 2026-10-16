"""Create an auditable manual calculation trace for one short test document."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage6_selection import Resources, bm25_pairs, build_features
from src.bm25 import BM25, seeded_tie_rank
from src.data import read_json, read_jsonl, write_json
from src.methods import descending_indices, expanded_weights, minmax, rerank_top_k, sigmoid_weight
from src.reranker import load_score_cache, pair_id
from src.sources import QuestionFeedbackIndex
from src.text import tokenize


SEED = 7
QID = "Q0100"
SCENARIO = "TPL"
METHODS = ("B1", "SOFT", "FINAL")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rankings(seed: int) -> list[dict]:
    path = ROOT / "results" / "runs" / f"seed_{seed:02d}_rankings.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def select_case() -> dict:
    candidates = []
    for seed in range(1, 11):
        rankings = read_rankings(seed)
        selected = {
            (row["qid"], row["scenario"], row["method"]): row
            for row in rankings
            if row["method"] in {"B1", "SOFT", "FINAL"}
        }
        observations = sorted({(qid, scenario) for qid, scenario, _ in selected})
        for qid, scenario in observations:
            b1 = selected[(qid, scenario, "B1")]
            soft = selected[(qid, scenario, "SOFT")]
            final = selected[(qid, scenario, "FINAL")]
            if b1["available_chunks"] >= 40:
                continue
            gold = b1["gold_chunk_id"]
            b1_rank = b1["ranking"].index(gold) + 1
            soft_rank = soft["ranking"].index(gold) + 1
            final_rank = final["ranking"].index(gold) + 1
            if soft_rank < b1_rank:
                candidates.append({
                    "seed": seed,
                    "qid": qid,
                    "scenario": scenario,
                    "doc_id": b1["doc_id"],
                    "available_chunks": b1["available_chunks"],
                    "gold_chunk_id": gold,
                    "rank_b1": b1_rank,
                    "rank_soft": soft_rank,
                    "rank_final": final_rank,
                    "gain": b1_rank - soft_rank,
                })
    chosen = max(
        candidates,
        key=lambda row: (row["gain"], -row["rank_soft"], -row["seed"], row["qid"], row["scenario"]),
    )
    expected = {"seed": SEED, "qid": QID, "scenario": SCENARIO}
    if any(chosen[key] != value for key, value in expected.items()):
        raise AssertionError(f"Selected case changed: {chosen}")
    chosen["eligible_candidates"] = len(candidates)
    return chosen


def bm25_term_detail(model: BM25, term: str, document_index: int) -> dict:
    frequency = model.doc_freqs[document_index].get(term, 0)
    document_frequency = sum(term in row for row in model.doc_freqs)
    raw_idf = (
        math.log(model.corpus_size - document_frequency + 0.5)
        - math.log(document_frequency + 0.5)
        if document_frequency
        else 0.0
    )
    denominator = frequency + model.k1 * (
        1.0 - model.b + model.b * model.doc_len[document_index] / model.avgdl
    )
    contribution = (
        model.idf.get(term, 0.0) * frequency * (model.k1 + 1.0) / denominator
        if denominator
        else 0.0
    )
    return {
        "term": term,
        "N": model.corpus_size,
        "df": document_frequency,
        "raw_idf": raw_idf,
        "idf_used": model.idf.get(term, 0.0),
        "tf": frequency,
        "dl": float(model.doc_len[document_index]),
        "avgdl": model.avgdl,
        "k1": model.k1,
        "b": model.b,
        "denominator": float(denominator),
        "contribution": float(contribution),
    }


def compact(text: str, limit: int = 150) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    value = re.sub(r"<[^>]+>", "[disamarkan]", value)
    value = value.replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def score_row(model: BM25, query_tokens: list[str], index: int, item_id: str) -> dict:
    contributions = model.explain(query_tokens, index)
    return {
        "item_id": item_id,
        "score": float(sum(contributions.values())),
        "document_length": int(model.doc_len[index]),
        "term_frequencies": {term: model.doc_freqs[index].get(term, 0) for term in dict.fromkeys(query_tokens)},
        "contributions": contributions,
    }


def reciprocal_rank(rank: int) -> float:
    return 1.0 / rank


def ndcg10(rank: int) -> float:
    return 0.0 if rank > 10 else 1.0 / math.log2(rank + 1.0)


def build_trace() -> dict:
    case = select_case()
    params = yaml.safe_load((ROOT / "results" / "params" / "seed_07.yaml").read_text(encoding="utf-8"))
    final_config = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text(encoding="utf-8"))
    resources = Resources(ROOT / "data" / "processed", ROOT / "models" / "embeddings")
    features = build_features(
        resources, ROOT / "data" / "processed", [SEED], "QS-BM25", partition="test"
    )
    feature = next(row for row in features if row["qid"] == QID and row["scenario"] == SCENARIO)
    pair = bm25_pairs([feature], int(params["m"]), int(params["r"]))[0]
    qa_raw, ext_raw = pair
    qa_norm, ext_norm = minmax(qa_raw), minmax(ext_raw)
    k = float(params["sigmoid"]["k"])
    tau = float(params["sigmoid"]["tau"])
    weight = sigmoid_weight(feature["s"], k, tau)
    soft_scores = weight * qa_norm + (1.0 - weight) * ext_norm
    b1_scores = feature["patient"].model.scores(feature["query_tokens"])
    b1_order = descending_indices(b1_scores)
    soft_order = descending_indices(soft_scores)

    split = read_json(ROOT / "data" / "processed" / "splits" / "seed_07.json")
    collection_qids = set(split["qids"]["collection"])
    collection_qa = [row for row in resources.qa if row["qid"] in collection_qids]
    question_index = QuestionFeedbackIndex(collection_qa, tie_seed=SEED * 1009)
    question_tokens = tokenize(feature["query"])
    question_scores = question_index.model.scores(question_tokens)
    question_ranking = seeded_tie_rank(
        question_scores,
        question_index.item_ids,
        seed=question_index.tie_seed,
        query_id=f"{QID}:{SCENARIO}",
        top_k=5,
    )
    if [item.item_id for item in question_ranking] != feature["q_feedback_qids"]:
        raise AssertionError("Question feedback reconstruction differs from Stage 7")

    question_idfs = []
    for term in dict.fromkeys(question_tokens):
        df = sum(term in row for row in question_index.model.doc_freqs)
        question_idfs.append({
            "term": term,
            "df": df,
            "idf": question_index.model.idf.get(term, 0.0),
        })
    question_top3 = [
        score_row(question_index.model, question_tokens, item.index, item.item_id)
        for item in question_ranking[:3]
    ]
    q_first = question_ranking[0].index
    q_term = max(
        question_tokens,
        key=lambda term: float(question_index.model.term_scores(term)[q_first]),
    )
    question_term_example = bm25_term_detail(question_index.model, q_term, q_first)

    patient = feature["patient"]
    b1_top3 = [
        score_row(patient.model, feature["query_tokens"], int(index), patient.item_ids[int(index)])
        for index in b1_order[:3]
    ]
    b1_first = int(b1_order[0])
    b1_term = max(
        feature["query_tokens"],
        key=lambda term: float(patient.model.term_scores(term)[b1_first]),
    )
    b1_term_example = bm25_term_detail(patient.model, b1_term, b1_first)

    m, repeat = int(params["m"]), int(params["r"])
    qa_weights = expanded_weights(feature["query_tokens"], feature["q_feedback_tokens"][:m], repeat)
    ext_weights = expanded_weights(feature["query_tokens"], feature["ext_feedback_tokens"][:m], repeat)
    soft_top3_indices = [int(index) for index in soft_order[:3]]
    normalization_rows = [{
        "chunk_id": patient.item_ids[index],
        "qa_raw": float(qa_raw[index]),
        "qa_normalized": float(qa_norm[index]),
        "ext_raw": float(ext_raw[index]),
        "ext_normalized": float(ext_norm[index]),
        "soft_score": float(soft_scores[index]),
    } for index in soft_top3_indices]

    score_cache = load_score_cache(ROOT / "models" / "reranker" / "stage6_scores.json")
    candidates = soft_order[: int(final_config["reranker"]["top_k"])]
    original_question = resources.qa_by_qid[QID]["question"]
    reranker_raw = np.asarray([
        score_cache[pair_id(original_question, patient.records[int(index)]["text"])]
        for index in candidates
    ], dtype=float)
    reranker_norm = minmax(reranker_raw)
    candidate_pipeline_norm = minmax(soft_scores[candidates])
    reranker_lambda = float(params["final_reranker"]["lambda"])
    combined = reranker_lambda * reranker_norm + (1.0 - reranker_lambda) * candidate_pipeline_norm
    final_order = rerank_top_k(soft_scores, candidates, reranker_raw, reranker_lambda)
    candidate_position = {int(index): pos for pos, index in enumerate(candidates)}
    final_top3 = []
    for index_value in final_order[:3]:
        index = int(index_value)
        pos = candidate_position[index]
        final_top3.append({
            "chunk_id": patient.item_ids[index],
            "soft_score": float(soft_scores[index]),
            "pipeline_candidate_normalized": float(candidate_pipeline_norm[pos]),
            "reranker_raw": float(reranker_raw[pos]),
            "reranker_normalized": float(reranker_norm[pos]),
            "combined_score": float(combined[pos]),
        })

    gold_index = feature["gold_index"]
    calculated_ranks = {
        "B1": int(np.flatnonzero(b1_order == gold_index)[0]) + 1,
        "SOFT": int(np.flatnonzero(soft_order == gold_index)[0]) + 1,
        "FINAL": int(np.flatnonzero(final_order == gold_index)[0]) + 1,
    }
    stored_rows = {
        row["method"]: row
        for row in read_rankings(SEED)
        if row["qid"] == QID and row["scenario"] == SCENARIO and row["method"] in METHODS
    }
    calculated_orders = {"B1": b1_order, "SOFT": soft_order, "FINAL": final_order}
    for method in METHODS:
        ids = [patient.item_ids[int(index)] for index in calculated_orders[method]]
        if ids != stored_rows[method]["ranking"]:
            raise AssertionError(f"{method} ranking differs from stored Stage 7 output")
        if calculated_ranks[method] != case[f"rank_{method.lower()}"]:
            raise AssertionError(f"{method} rank differs from selected case")

    with (ROOT / "results" / "stage8" / "robustness_summary.csv").open(encoding="utf-8") as handle:
        robustness_rows = list(csv.DictReader(handle))
    ri_source = next(
        row for row in robustness_rows if row["method"] == "FINAL" and row["scenario"] == "average"
    )
    wins, ties, losses, total = (
        int(ri_source["wins"]), int(ri_source["ties"]), int(ri_source["losses"]), int(ri_source["n_total"])
    )
    ri_calculated = (wins - losses) / total
    if not math.isclose(ri_calculated, float(ri_source["robustness_index_pooled"]), abs_tol=1e-15):
        raise AssertionError("RI reconstruction failed")

    return {
        "selection": case,
        "question": {
            "text": feature["query"],
            "tokens_before_stopword_removal": question_tokens,
            "tokens_after_stopword_removal": feature["query_tokens"],
        },
        "question_similarity": {
            "corpus_size": question_index.model.corpus_size,
            "avgdl": question_index.model.avgdl,
            "idf_table": question_idfs,
            "top3": question_top3,
            "term_example": question_term_example,
            "s": float(feature["s"]),
        },
        "feedback": {
            "m": m,
            "repeat": repeat,
            "qa": [{
                "qid": qid,
                "score": score,
                "excerpt": compact(text),
                "token_count": len(tokens),
            } for qid, score, text, tokens in zip(
                feature["q_feedback_qids"][:m],
                feature["q_feedback_scores"][:m],
                feature["q_feedback_texts"][:m],
                feature["q_feedback_tokens"][:m],
            )],
            "external": [{
                "chunk_id": chunk_id,
                "excerpt": compact(text),
                "token_count": len(tokens),
            } for chunk_id, text, tokens in zip(
                feature["ext_feedback_ids"][:m],
                feature["ext_feedback_texts"][:m],
                feature["ext_feedback_tokens"][:m],
            )],
            "qa_weights": dict(sorted(qa_weights.items())),
            "external_weights": dict(sorted(ext_weights.items())),
        },
        "baseline": {
            "corpus_size": patient.model.corpus_size,
            "avgdl": patient.model.avgdl,
            "top3": b1_top3,
            "term_example": b1_term_example,
            "gold_score": float(b1_scores[gold_index]),
            "gold_rank": calculated_ranks["B1"],
        },
        "normalization": {
            "qa_min": float(qa_raw.min()),
            "qa_max": float(qa_raw.max()),
            "ext_min": float(ext_raw.min()),
            "ext_max": float(ext_raw.max()),
            "rows": normalization_rows,
        },
        "soft": {
            "k": k,
            "tau": tau,
            "s": float(feature["s"]),
            "weight": weight,
            "rows": normalization_rows,
            "gold_rank": calculated_ranks["SOFT"],
        },
        "reranker": {
            "model_id": "BAAI/bge-reranker-v2-m3",
            "top_k": len(candidates),
            "lambda": reranker_lambda,
            "raw_min": float(reranker_raw.min()),
            "raw_max": float(reranker_raw.max()),
            "pipeline_candidate_min": float(soft_scores[candidates].min()),
            "pipeline_candidate_max": float(soft_scores[candidates].max()),
            "top3": final_top3,
            "gold_rank": calculated_ranks["FINAL"],
        },
        "metrics": {
            method: {
                "rank": rank,
                "reciprocal_rank": reciprocal_rank(rank),
                "ndcg_at_10": ndcg10(rank),
            }
            for method, rank in calculated_ranks.items()
        },
        "robustness_example": {
            "method": "FINAL vs B1",
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "n": total,
            "ri": ri_calculated,
        },
    }


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def terms(values: dict[str, float], limit: int = 4) -> str:
    selected = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return "; ".join(f"{term}={fmt(score)}" for term, score in selected if score != 0) or "semua 0"


def render_report(trace: dict, trace_path: Path) -> str:
    selection = trace["selection"]
    question = trace["question"]
    qs = trace["question_similarity"]
    qterm = qs["term_example"]
    base = trace["baseline"]
    bterm = base["term_example"]
    norm = trace["normalization"]
    soft = trace["soft"]
    reranker = trace["reranker"]
    metrics = trace["metrics"]
    ri = trace["robustness_example"]

    lines = [
        "# Tahap 12 — Jejak Perhitungan Manual",
        "",
        "## Pemilihan contoh",
        "",
        f"Contoh dipilih secara deterministik sebagai kenaikan B1→SOFT terbesar dari {selection['eligible_candidates']} "
        f"observasi test yang dokumennya memiliki kurang dari 40 chunk. Hasilnya adalah seed **{selection['seed']}**, "
        f"`{selection['qid']}` skenario **{selection['scenario']}** pada `{selection['doc_id']}` ({selection['available_chunks']} chunk). "
        f"Chunk gold `{selection['gold_chunk_id']}` naik dari rank {selection['rank_b1']} menjadi {selection['rank_soft']}; "
        f"FINAL menempatkannya pada rank {selection['rank_final']}.",
        "",
        "## 1. Tokenisasi pertanyaan",
        "",
        f"> {question['text']}",
        "",
        f"- Sebelum penghapusan stopword: `{question['tokens_before_stopword_removal']}`",
        f"- Setelah penghapusan stopword (`q'`): `{question['tokens_after_stopword_removal']}`",
        "",
        "Tokenisasi memakai lowercase dan regex `\\w+`. Daftar Sastrawi ditambah `apa`, `bagaimana`, `yang`, `ini`, dan `pasien`.",
        "",
        "## 2. Question similarity QS-BM25",
        "",
        f"Indeks pertanyaan koleksi berisi N={qs['corpus_size']} pertanyaan dengan `avgdl={fmt(qs['avgdl'])}`. "
        "QS memakai token sebelum stopword dihapus. IDF negatif mengikuti implementasi `rank_bm25`: "
        "nilainya diganti dengan `epsilon × average_IDF`, sehingga beberapa term umum memiliki IDF positif yang sama.",
        "",
        "| Term | df | IDF yang dipakai |",
        "|---|---:|---:|",
    ]
    for row in qs["idf_table"]:
        lines.append(f"| `{row['term']}` | {row['df']} | {fmt(row['idf'])} |")
    lines.extend(["", "Tiga pertanyaan feedback teratas:", "", "| Rank | QID | Panjang | Skor | Kontribusi term terbesar |", "|---:|---|---:|---:|---|"])
    for rank, row in enumerate(qs["top3"], 1):
        lines.append(f"| {rank} | `{row['item_id']}` | {row['document_length']} | {fmt(row['score'])} | {terms(row['contributions'])} |")
    lines.extend([
        "",
        f"Contoh lengkap term `{qterm['term']}` pada pertanyaan rank 1:",
        "",
        f"- IDF mentah = `ln(({qterm['N']} - {qterm['df']} + 0,5) / ({qterm['df']} + 0,5)) = {fmt(qterm['raw_idf'])}`; IDF yang dipakai = `{fmt(qterm['idf_used'])}`.",
        f"- tf={qterm['tf']}, dl={fmt(qterm['dl'], 0)}, avgdl={fmt(qterm['avgdl'])}, k1={fmt(qterm['k1'], 1)}, b={fmt(qterm['b'], 2)}.",
        f"- Penyebut = `tf + k1 × (1-b + b × dl/avgdl) = {fmt(qterm['denominator'])}`.",
        f"- Kontribusi = `IDF × tf × (k1+1) / penyebut = {fmt(qterm['contribution'])}`.",
        f"- Skor similarity maksimum `s = {fmt(qs['s'])}` (jumlah kontribusi seluruh term query pada pertanyaan rank 1).",
        "",
        "## 3. Feedback dan susunan query ekspansi",
        "",
        f"Parameter seed 7 adalah `m={trace['feedback']['m']}` dan `r={trace['feedback']['repeat']}`. Lima jawaban feedback QA:",
        "",
        "| Rank | QID | Skor QS | Token | Kutipan dipersingkat |",
        "|---:|---|---:|---:|---|",
    ])
    for rank, row in enumerate(trace["feedback"]["qa"], 1):
        lines.append(f"| {rank} | `{row['qid']}` | {fmt(row['score'])} | {row['token_count']} | {row['excerpt']} |")
    lines.extend(["", "Lima chunk feedback eksternal:", "", "| Rank | Chunk | Token | Kutipan dipersingkat |", "|---:|---|---:|---|"])
    for rank, row in enumerate(trace["feedback"]["external"], 1):
        lines.append(f"| {rank} | `{row['chunk_id']}` | {row['token_count']} | {row['excerpt']} |")
    q_base = " ".join(question["tokens_after_stopword_removal"])
    lines.extend([
        "",
        f"Susunan tepat secara konseptual: `q_QA = [{q_base}] × 3 + jawaban Q0651 + Q0376 + Q0209 + Q0296 + Q0024`; "
        f"`q_EXT = [{q_base}] × 3 + chunk " + " + ".join(row["chunk_id"] for row in trace["feedback"]["external"]) + "`.",
        "",
        "Implementasi BM25 memakai bag-of-words berbobot: setiap kemunculan term query dasar ditambah bobot 3, lalu seluruh kemunculan token feedback ditambahkan dengan bobot 1. Representasi lengkap tersimpan di artefak privat `results/stage12/manual_trace.json`.",
        "",
        "## 4. BM25 baseline B1",
        "",
        f"Indeks pasien mempunyai N={base['corpus_size']} chunk dan `avgdl={fmt(base['avgdl'])}`. Tiga chunk teratas:",
        "",
        "| Rank | Chunk | Panjang | Skor B1 | Kontribusi term |",
        "|---:|---|---:|---:|---|",
    ])
    for rank, row in enumerate(base["top3"], 1):
        lines.append(f"| {rank} | `{row['item_id']}` | {row['document_length']} | {fmt(row['score'])} | {terms(row['contributions'], 5)} |")
    lines.extend([
        "",
        f"Contoh lengkap term `{bterm['term']}` pada chunk rank 1:",
        "",
        f"- IDF mentah = `ln(({bterm['N']} - {bterm['df']} + 0,5) / ({bterm['df']} + 0,5)) = {fmt(bterm['raw_idf'])}`; IDF dipakai `{fmt(bterm['idf_used'])}`.",
        f"- tf={bterm['tf']}, dl={fmt(bterm['dl'], 0)}, avgdl={fmt(bterm['avgdl'])}, penyebut={fmt(bterm['denominator'])}.",
        f"- Kontribusi = `{fmt(bterm['idf_used'])} × {bterm['tf']} × ({fmt(bterm['k1'], 1)}+1) / {fmt(bterm['denominator'])} = {fmt(bterm['contribution'])}`.",
        f"Chunk gold memiliki skor B1 `{fmt(base['gold_score'])}` dan rank **{base['gold_rank']}**.",
        "",
        "## 5. Normalisasi min–max",
        "",
        f"Normalisasi dilakukan atas seluruh {selection['available_chunks']} chunk: QA min/max = `{fmt(norm['qa_min'])}`/`{fmt(norm['qa_max'])}`, "
        f"EXT min/max = `{fmt(norm['ext_min'])}`/`{fmt(norm['ext_max'])}`. Rumusnya `x̂=(x-min)/(max-min)`.",
        "",
        "| Chunk (top-3 SOFT) | QA raw | QA norm | EXT raw | EXT norm |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in norm["rows"]:
        lines.append(f"| `{row['chunk_id']}` | {fmt(row['qa_raw'])} | {fmt(row['qa_normalized'])} | {fmt(row['ext_raw'])} | {fmt(row['ext_normalized'])} |")
    lines.extend([
        "",
        "## 6. Bobot sigmoid dan skor SOFT",
        "",
        f"Dengan `k={fmt(soft['k'], 1)}`, `τ={fmt(soft['tau'])}`, dan `s={fmt(soft['s'])}`:",
        "",
        f"`w(s) = 1 / (1 + exp(-{fmt(soft['k'], 1)} × ({fmt(soft['s'])} - {fmt(soft['tau'])}))) = {fmt(soft['weight'])}`.",
        "",
        "| Chunk | Perhitungan SOFT | Skor SOFT |",
        "|---|---|---:|",
    ])
    for row in soft["rows"]:
        lines.append(
            f"| `{row['chunk_id']}` | `{fmt(soft['weight'])}×{fmt(row['qa_normalized'])} + "
            f"{fmt(1-soft['weight'])}×{fmt(row['ext_normalized'])}` | {fmt(row['soft_score'])} |"
        )
    lines.extend([
        "",
        f"Chunk gold menjadi rank **{soft['gold_rank']}** pada SOFT.",
        "",
        "## 7. Reranker FINAL",
        "",
        f"FINAL mererank top-{reranker['top_k']} kandidat SOFT dengan `{reranker['model_id']}` dan λ={fmt(reranker['lambda'], 2)}. "
        "Skor pipeline dan reranker dinormalisasi min–max lagi di dalam 20 kandidat, lalu digabungkan sebagai "
        "`FINAL = λ × reranker_norm + (1-λ) × pipeline_norm`.",
        "",
        f"Rentang kandidat: pipeline `{fmt(reranker['pipeline_candidate_min'])}`–`{fmt(reranker['pipeline_candidate_max'])}`; "
        f"reranker raw `{fmt(reranker['raw_min'])}`–`{fmt(reranker['raw_max'])}`.",
        "",
        "| Rank FINAL | Chunk | Pipeline norm | Reranker raw | Reranker norm | Skor gabungan |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(reranker["top3"], 1):
        lines.append(f"| {rank} | `{row['chunk_id']}` | {fmt(row['pipeline_candidate_normalized'])} | {fmt(row['reranker_raw'])} | {fmt(row['reranker_normalized'])} | {fmt(row['combined_score'])} |")
    lines.extend([
        "",
        f"Chunk gold tetap pada rank **{reranker['gold_rank']}** setelah reranking.",
        "",
        "## 8. Dampak pada metrik",
        "",
        "Dengan satu gold ketat, `RR=1/rank`. Jika rank≤10, `nDCG@10=1/log2(rank+1)`; jika rank>10 nilainya 0.",
        "",
        "| Metode | Rank gold | Reciprocal rank | nDCG@10 |",
        "|---|---:|---:|---:|",
    ])
    for method in METHODS:
        row = metrics[method]
        lines.append(f"| {method} | {row['rank']} | {fmt(row['reciprocal_rank'])} | {fmt(row['ndcg_at_10'])} |")
    lines.extend([
        "",
        "Jadi, untuk contoh ini SOFT menaikkan RR dari 0,071429 menjadi 1,000000 dan nDCG@10 dari 0 menjadi 1. Reranker tidak mengubah rank gold lebih lanjut.",
        "",
        "## 9. Contoh Robustness Index",
        "",
        f"Dari tabel pooled FINAL vs B1 Tahap 8: win={ri['wins']}, tie={ri['ties']}, loss={ri['losses']}, N={ri['n']}. Tie tetap masuk penyebut tetapi tidak masuk pembilang:",
        "",
        f"`RI = (N⁺ - N⁻) / N = ({ri['wins']} - {ri['losses']}) / {ri['n']} = {fmt(ri['ri'])}`.",
        "",
        "Nilai positif berarti FINAL lebih sering memperbaiki daripada memperburuk rank gold relatif terhadap B1.",
        "",
        "## Verifikasi",
        "",
        "Semua ranking B1, SOFT, dan FINAL direkonstruksi dari fitur serta cache skor, kemudian dicocokkan persis dengan ranking beku Tahap 7.",
        f"SHA-256 artefak trace privat: `{file_sha256(trace_path)}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    output_dir = ROOT / "results" / "stage12"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "manual_trace.json"
    trace = build_trace()
    write_json(trace_path, trace)
    report_path = ROOT / "reports" / "12_manual_trace.md"
    report_path.write_text(render_report(trace, trace_path), encoding="utf-8")
    print(json.dumps({
        "case": trace["selection"],
        "metrics": trace["metrics"],
        "ri": trace["robustness_example"],
        "trace_sha256": file_sha256(trace_path),
        "report": str(report_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
