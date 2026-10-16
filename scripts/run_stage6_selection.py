"""Select question similarity and chunk retriever components on dev seeds 1--3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bm25 import seeded_tie_rank
from src.data import QUESTION_TYPE_ORDER, read_json, read_jsonl, write_json
from src.dense import encode_e5_transformers, validate_embeddings
from src.methods import descending_indices, expanded_weights, minmax, rank_of, rrf_scores, sigmoid_weight
from src.semantic import cosine_scores, normalized_average, rocchio_vector, rrf_pair
from src.sources import ExternalFeedbackIndex, PatientBM25Index, QuestionFeedbackIndex
from src.text import tokenize


SCENARIOS = ("TPL", "NEW", "PARA", "PNEW")
QS_METHODS = ("QS-BM25", "QS-EMB", "QS-HYB-MM", "QS-HYB-RRF")
ROCCHIO_VALUES = (0.25, 0.5, 0.75, 1.0)
MODEL_ID = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_by_scenario(rows: list[dict]) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(float(row["rr"]))
    result = {scenario: float(np.mean(grouped[scenario])) for scenario in SCENARIOS}
    result["mean"] = float(np.mean(list(result.values())))
    return result


class Resources:
    def __init__(self, processed: Path, embeddings: Path) -> None:
        self.qa = read_jsonl(processed / "qa.jsonl")
        self.chunks = read_jsonl(processed / "chunks.jsonl")
        self.qa_by_qid = {row["qid"]: row for row in self.qa}
        self.chunk_by_id = {row["chunk_id"]: row for row in self.chunks}
        ids = read_json(embeddings / "ids.json")
        self.qid_to_index = {value: index for index, value in enumerate(ids["qids"])}
        self.chunk_to_index = {value: index for index, value in enumerate(ids["chunk_ids"])}
        self.question_vectors = validate_embeddings(np.load(embeddings / "questions.npy"), rows=len(self.qa))
        self.answer_vectors = validate_embeddings(np.load(embeddings / "answers.npy"), rows=len(self.qa))
        self.chunk_vectors = validate_embeddings(np.load(embeddings / "chunks.npy"), rows=len(self.chunks))
        paraphrase_rows = read_json(embeddings / "paraphrase_ids.json")
        paraphrase_vectors = validate_embeddings(
            np.load(embeddings / "paraphrases.npy"), rows=len(paraphrase_rows),
        )
        self.paraphrase_by_text = {
            row["text"]: paraphrase_vectors[index] for index, row in enumerate(paraphrase_rows)
        }
        self.patient = {}
        by_doc = defaultdict(list)
        for row in self.chunks:
            by_doc[row["doc_id"]].append(row)
        for doc_id, rows in by_doc.items():
            index = PatientBM25Index(rows)
            self.patient[doc_id] = {
                "bm25": index,
                "vectors": self.chunk_vectors[[self.chunk_to_index[item] for item in index.item_ids]],
            }

    def query_vector(self, row: dict) -> np.ndarray:
        original = self.qa_by_qid[row["qid"]]
        if row["query"] == original["question"]:
            return self.question_vectors[self.qid_to_index[row["qid"]]]
        try:
            return self.paraphrase_by_text[row["query"]]
        except KeyError as error:
            raise KeyError(f"Embedding query tidak ditemukan: {row['qid']} {row['scenario']}") from error


def question_scores(
    method: str, query: str, query_vector: np.ndarray, index: QuestionFeedbackIndex,
    vectors: np.ndarray, *, seed: int, query_id: str,
) -> np.ndarray:
    bm25 = index.model.scores(tokenize(query))
    if method == "QS-BM25":
        return bm25
    dense = cosine_scores(query_vector, vectors)
    if method == "QS-EMB":
        return dense
    if method == "QS-HYB-MM":
        return normalized_average(bm25, dense)
    if method == "QS-HYB-RRF":
        bm25_order = [x.index for x in seeded_tie_rank(
            bm25, index.item_ids, seed=seed, query_id=f"{query_id}:bm25",
        )]
        dense_order = [x.index for x in seeded_tie_rank(
            dense, index.item_ids, seed=seed, query_id=f"{query_id}:dense",
        )]
        return rrf_scores([bm25_order, dense_order], len(index.item_ids), k=60)
    raise ValueError(f"Metode QS tidak dikenal: {method}")


def build_features(
    resources: Resources, processed: Path, seeds: list[int], qs_method: str,
    *, partition: str = "dev",
) -> list[dict]:
    if partition not in {"dev", "test"}:
        raise ValueError(f"Partisi tidak dikenal: {partition}")
    features = []
    for seed in seeds:
        split = read_json(processed / "splits" / f"seed_{seed:02d}.json")
        collection_qids = set(split["qids"]["collection"])
        collection_docs = set(split["documents"]["collection"])
        collection_qa = [row for row in resources.qa if row["qid"] in collection_qids]
        collection_chunks = [row for row in resources.chunks if row["doc_id"] in collection_docs]
        q_indexes = {}
        q_vectors = {}
        for exclusion in (None, *QUESTION_TYPE_ORDER):
            rows = [row for row in collection_qa if exclusion is None or row["question_type"] != exclusion]
            index = QuestionFeedbackIndex(rows, tie_seed=seed * 1009)
            q_indexes[exclusion] = index
            q_vectors[exclusion] = resources.question_vectors[
                [resources.qid_to_index[qid] for qid in index.item_ids]
            ]
        external = ExternalFeedbackIndex(collection_chunks)
        external_cache = {}
        scenario_rows = [
            row for row in read_jsonl(processed / "scenarios" / f"seed_{seed:02d}.jsonl")
            if row["partition"] == partition
        ]
        for row in scenario_rows:
            exclusion = row["question_type"] if row["scenario"] in {"NEW", "PNEW"} else None
            q_index = q_indexes[exclusion]
            query_vector = resources.query_vector(row)
            scores = question_scores(
                qs_method, row["query"], query_vector, q_index, q_vectors[exclusion],
                seed=q_index.tie_seed, query_id=f"{row['qid']}:{row['scenario']}",
            )
            ranking = seeded_tie_rank(
                scores, q_index.item_ids, seed=q_index.tie_seed,
                query_id=f"{row['qid']}:{row['scenario']}", top_k=5,
            )
            q_feedback = [q_index.records[item.index] for item in ranking]
            q_tokens = tokenize(row["query"], remove_stopwords=True)
            ext_key = row["query"]
            if ext_key not in external_cache:
                ext_scores = external.model.scores(q_tokens)
                order = descending_indices(ext_scores)[:5]
                external_cache[ext_key] = [external.records[int(index)] for index in order]
            ext_feedback = external_cache[ext_key]
            patient = resources.patient[row["doc_id"]]
            pindex = patient["bm25"]
            features.append({
                **row,
                "patient": pindex,
                "patient_vectors": patient["vectors"],
                "gold_index": pindex.item_ids.index(row["gold_chunk_id"]),
                "query_tokens": q_tokens,
                "query_vector": query_vector,
                "q_feedback_tokens": [
                    tokenize(" ".join(item["answer_texts"]), remove_stopwords=True) for item in q_feedback
                ],
                "q_feedback_texts": [" ".join(item["answer_texts"]) for item in q_feedback],
                "q_feedback_vectors": resources.answer_vectors[
                    [resources.qid_to_index[item["qid"]] for item in q_feedback]
                ],
                "q_feedback_qids": [item["qid"] for item in q_feedback],
                "q_feedback_scores": [float(item.score) for item in ranking],
                "ext_feedback_tokens": [
                    tokenize(item["text"], remove_stopwords=True) for item in ext_feedback
                ],
                "ext_feedback_texts": [item["text"] for item in ext_feedback],
                "ext_feedback_vectors": resources.chunk_vectors[
                    [resources.chunk_to_index[item["chunk_id"]] for item in ext_feedback]
                ],
                "ext_feedback_ids": [item["chunk_id"] for item in ext_feedback],
                "s": float(ranking[0].score),
            })
    return features


def bm25_pairs(features: list[dict], m: int, repeat: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for feature in features:
        qa_weights = expanded_weights(
            feature["query_tokens"], feature["q_feedback_tokens"][:m], repeat,
        )
        ext_weights = expanded_weights(
            feature["query_tokens"], feature["ext_feedback_tokens"][:m], repeat,
        )
        pairs.append((
            feature["patient"].model.weighted_scores(qa_weights),
            feature["patient"].model.weighted_scores(ext_weights),
        ))
    return pairs


def expanded_text(feature: dict, source: str, m: int, repeat: int) -> str:
    query = " ".join(feature["query_tokens"])
    feedback = feature[f"{source}_feedback_texts"][:m]
    return " ".join([query] * repeat + feedback)


def full_query_vectors(
    features: list[dict], cache_dir: Path, model_cache: Path, m: int, repeat: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    texts = [expanded_text(feature, source, m, repeat) for feature in features for source in ("q", "ext")]
    hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    vector_path = cache_dir / "stage6_full_dev.npy"
    id_path = cache_dir / "stage6_full_dev_ids.json"
    if vector_path.exists() and id_path.exists() and read_json(id_path) == hashes:
        values = validate_embeddings(np.load(vector_path), rows=len(texts))
    else:
        unique_texts = list(dict.fromkeys(texts))
        print(f"Dense full-text: {len(texts)} inputs, {len(unique_texts)} unique", flush=True)
        unique_values = encode_e5_transformers(
            unique_texts, model_id=MODEL_ID, revision=MODEL_REVISION, cache_folder=model_cache,
            kind="query", batch_size=8, progress=True,
        )
        lookup = {text: unique_values[index] for index, text in enumerate(unique_texts)}
        values = np.asarray([lookup[text] for text in texts], dtype=np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(vector_path, values)
        write_json(id_path, hashes)
    return [(values[index], values[index + 1]) for index in range(0, len(values), 2)]


def dense_pairs(
    features: list[dict], *, mode: str, full_vectors: list[tuple[np.ndarray, np.ndarray]] | None = None,
    alpha: float | None = None, beta: float | None = None, m: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for number, feature in enumerate(features):
        if mode == "full":
            if full_vectors is None:
                raise ValueError("Embedding full-text belum tersedia")
            qa_vector, ext_vector = full_vectors[number]
        elif mode == "rocchio":
            qa_vector = rocchio_vector(
                feature["query_vector"], feature["q_feedback_vectors"][:m], alpha=alpha, beta=beta,
            )
            ext_vector = rocchio_vector(
                feature["query_vector"], feature["ext_feedback_vectors"][:m], alpha=alpha, beta=beta,
            )
        else:
            raise ValueError(mode)
        result.append((
            cosine_scores(qa_vector, feature["patient_vectors"]),
            cosine_scores(ext_vector, feature["patient_vectors"]),
        ))
    return result


def hybrid_pairs(
    bm25: list[tuple[np.ndarray, np.ndarray]], dense: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (rrf_pair(bq, dq, k=60), rrf_pair(be, de, k=60))
        for (bq, be), (dq, de) in zip(bm25, dense)
    ]


def tune_soft(
    features: list[dict], pairs: list[tuple[np.ndarray, np.ndarray]], identity: dict,
) -> tuple[dict, list[dict]]:
    normalized = [(minmax(qa), minmax(ext)) for qa, ext in pairs]
    s_values = np.asarray([feature["s"] for feature in features])
    taus = sorted(set(float(value) for value in np.percentile(s_values, np.arange(5, 100, 5))))
    rows = []
    for k in (0.5, 1.0, 2.0, 4.0, 8.0):
        for tau in taus:
            per_query = []
            for feature, (qa_score, ext_score) in zip(features, normalized):
                weight = sigmoid_weight(feature["s"], k, tau)
                score = weight * qa_score + (1.0 - weight) * ext_score
                per_query.append({
                    "scenario": feature["scenario"],
                    "rr": 1.0 / rank_of(score, feature["gold_index"]),
                })
            rows.append({**identity, "k": k, "tau": tau, **mean_by_scenario(per_query)})
    return max(rows, key=lambda row: (row["mean"], -row["k"], -row["tau"])), rows


def eligible(candidate: dict, previous: dict) -> bool:
    return all(candidate[scenario] >= previous[scenario] - 0.01 for scenario in SCENARIOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--model-cache", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--results", type=Path, default=Path("results/stage6"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(Path("configs/stage5.yaml").read_text(encoding="utf-8"))
    m, repeat = int(base_config["base"]["m"]), int(base_config["base"]["r"])
    resources = Resources(args.processed, args.embeddings)
    started = time.perf_counter()

    qs_summaries, qs_grids, features_by_qs, bm25_by_qs = [], [], {}, {}
    for method in QS_METHODS:
        print(f"Building {method} features", flush=True)
        features = build_features(resources, args.processed, args.seeds, method)
        pairs = bm25_pairs(features, m, repeat)
        best, grid = tune_soft(features, pairs, {"qs_method": method, "retriever": "R-BM25"})
        qs_summaries.append(best); qs_grids.extend(grid)
        features_by_qs[method] = features; bm25_by_qs[method] = pairs
        print(json.dumps(best), flush=True)

    previous_qs = next(row for row in qs_summaries if row["qs_method"] == "QS-BM25")
    qs_candidates = [row for row in qs_summaries if eligible(row, previous_qs)]
    selected_qs = max(qs_candidates, key=lambda row: row["mean"])
    features = features_by_qs[selected_qs["qs_method"]]
    bm25 = bm25_by_qs[selected_qs["qs_method"]]
    write_csv(args.results / "qs_candidates.csv", qs_summaries)
    write_csv(args.results / "qs_grid.csv", qs_grids)

    print(f"Encoding dense full-text candidates for {selected_qs['qs_method']}", flush=True)
    full_vectors = full_query_vectors(features, args.embeddings, args.model_cache, m, repeat)
    dense_full = dense_pairs(features, mode="full", full_vectors=full_vectors, m=m)
    retriever_pairs = {"R-BM25": bm25, "R-DENSE-FULL": dense_full, "R-HYB-FULL": hybrid_pairs(bm25, dense_full)}
    for alpha in ROCCHIO_VALUES:
        for beta in ROCCHIO_VALUES:
            label = f"R-DENSE-ROCCHIO-A{alpha:g}-B{beta:g}"
            dense = dense_pairs(features, mode="rocchio", alpha=alpha, beta=beta, m=m)
            retriever_pairs[label] = dense
            retriever_pairs[label.replace("R-DENSE", "R-HYB")] = hybrid_pairs(bm25, dense)

    retriever_summaries, retriever_grids = [], []
    for method, pairs in retriever_pairs.items():
        print(f"Tuning {method}", flush=True)
        best, grid = tune_soft(
            features, pairs, {"qs_method": selected_qs["qs_method"], "retriever": method},
        )
        retriever_summaries.append(best); retriever_grids.extend(grid)
        print(json.dumps(best), flush=True)
    previous_retriever = next(row for row in retriever_summaries if row["retriever"] == "R-BM25")
    retriever_candidates = [row for row in retriever_summaries if eligible(row, previous_retriever)]
    selected_retriever = max(retriever_candidates, key=lambda row: row["mean"])
    write_csv(args.results / "retriever_candidates.csv", retriever_summaries)
    write_csv(args.results / "retriever_grid.csv", retriever_grids)

    selected = {
        "selection_seeds": args.seeds,
        "m": m,
        "r": repeat,
        "question_similarity": selected_qs["qs_method"],
        "retriever": selected_retriever["retriever"],
        "k": float(selected_retriever["k"]),
        "tau": float(selected_retriever["tau"]),
        "metrics": {key: float(selected_retriever[key]) for key in (*SCENARIOS, "mean")},
        "reranker": {"status": "pending", "top_k": 20, "rrf_k": 60},
        "selection_constraint": "no_scenario_drop_gt_0.01_vs_previous_step",
    }
    Path("configs/stage6_pre_reranker.yaml").write_text(
        yaml.safe_dump(selected, sort_keys=False), encoding="utf-8",
    )
    audit = {
        "features": len(features), "elapsed_seconds": time.perf_counter() - started,
        "selected_qs": selected_qs, "selected_retriever": selected_retriever,
        "eligible_qs": [row["qs_method"] for row in qs_candidates],
        "eligible_retrievers": [row["retriever"] for row in retriever_candidates],
    }
    write_json(args.results / "selection_pre_reranker.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
