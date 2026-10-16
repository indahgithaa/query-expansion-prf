"""Build the Kaggle notebook used for Stage 11 generation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


CELLS = [
    markdown(
        """# Stage 11 — Grounded generation

Private Kaggle execution for seed-1 TPL/PARA contexts. Run first with `TEST_MODE = True`, inspect the 5-QA sample, then set it to `False` and run all cells again. Cached answers are reused."""
    ),
    code(
        """# Environment gate: this must be the first executable cell.
import platform
import sys
import urllib.request

import torch

print("Python:", sys.version)
print("Platform:", platform.platform())
gpu_count = torch.cuda.device_count()
gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", gpu_count)
print("GPU names:", gpu_names)

if not torch.cuda.is_available():
    raise RuntimeError("GPU MATI: buka Notebook options dan pilih accelerator GPU T4 x2.")
if gpu_count < 2 or not all("T4" in name for name in gpu_names[:2]):
    raise RuntimeError(f"GPU T4 x2 tidak aktif; terdeteksi {gpu_names}. Ubah accelerator sebelum lanjut.")

try:
    with urllib.request.urlopen("https://huggingface.co", timeout=15) as response:
        print("Internet check:", response.status, response.url)
except Exception as exc:
    raise RuntimeError("INTERNET MATI: aktifkan Internet pada Notebook options.") from exc

print("Environment gate: PASS")"""
    ),
    code(
        """# Candidate pins; retain them only after a successful Kaggle test run.
%pip install -q --upgrade \\
    "transformers==4.57.1" \\
    "bitsandbytes==0.48.2" \\
    "accelerate==1.11.0" \\
    "bert-score==0.3.13" \\
    "rouge-score==0.1.2" \\
    "sentencepiece==0.2.1"

import accelerate, bitsandbytes, transformers
print("transformers", transformers.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("accelerate", accelerate.__version__)"""
    ),
    code(
        """from pathlib import Path
import gc
import json
import time

TEST_MODE = True          # After inspecting 5 QA, change to False and rerun all.
TEST_QA_LIMIT = 5
BATCH_SIZE = 4
MAX_NEW_TOKENS = 256
PRIMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_PATH = Path("/kaggle/working/gen_outputs.jsonl")
METRICS_PATH = Path("/kaggle/working/gen_metrics.jsonl")

matches = sorted(Path("/kaggle/input").rglob("gen_inputs.jsonl"))
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one gen_inputs.jsonl, found: {matches}")
INPUT_PATH = matches[0]
rows = [json.loads(line) for line in INPUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

expected_configs = {"B1", "RM3", "SOFT", "FINAL", "ORACLE"}
assert len(rows) == 1730, len(rows)
assert {row["configuration"] for row in rows} == expected_configs
assert {row["scenario"] for row in rows} == {"TPL", "PARA"}
assert len({(row["id"], row["configuration"]) for row in rows}) == len(rows)

if TEST_MODE:
    selected_ids = list(dict.fromkeys(row["id"] for row in rows))[:TEST_QA_LIMIT]
    selected = [row for row in rows if row["id"] in selected_ids]
else:
    selected = rows
print("Input:", INPUT_PATH)
print("Mode:", "TEST" if TEST_MODE else "FULL", "rows:", len(selected), "QA:", len({r['id'] for r in selected}))"""
    ),
    code(
        '''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if "model" in globals():
    del model
    gc.collect()
    torch.cuda.empty_cache()

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def load_model(model_id):
    print("Loading", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model

try:
    active_model_id = PRIMARY_MODEL
    tokenizer, model = load_model(active_model_id)
except Exception as primary_error:
    print("Primary model failed:", repr(primary_error))
    gc.collect()
    torch.cuda.empty_cache()
    active_model_id = FALLBACK_MODEL
    tokenizer, model = load_model(active_model_id)

model_revision = getattr(model.config, "_commit_hash", None)
print("Active model:", active_model_id)
print("Model revision:", model_revision)
print("Device map:", getattr(model, "hf_device_map", None))'''
    ),
    code(
        '''PROMPT_TEMPLATE = """Anda adalah asisten dietisien. Jawab pertanyaan berdasarkan kutipan laporan
asuhan gizi pasien berikut. Gunakan hanya informasi dari kutipan. Jika
informasinya tidak ada, jawab \"Informasi tidak ditemukan dalam laporan.\"

Kutipan:
[K1] {chunk_1}
[K2] {chunk_2}
[K3] {chunk_3}

Pertanyaan: {question}
Jawaban:"""

def cache_key(row):
    return row["configuration"], row["id"]

def load_cache(path):
    if not path.exists():
        return {}
    cached = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            cached[cache_key(item)] = item
    return cached

def append_rows(path, batch):
    with path.open("a", encoding="utf-8") as handle:
        for item in batch:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\\n")

cache = load_cache(OUTPUT_PATH)
pending = [row for row in selected if cache_key(row) not in cache]
print("Cached:", len(cache), "pending:", len(pending))

for start in range(0, len(pending), BATCH_SIZE):
    batch = pending[start:start + BATCH_SIZE]
    raw_prompts = [PROMPT_TEMPLATE.format(**row) for row in batch]
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in raw_prompts
    ]
    encoded = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    first_device = next(model.parameters()).device
    encoded = {name: tensor.to(first_device) for name, tensor in encoded.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_tokens = generated[:, encoded["input_ids"].shape[1]:]
    answers = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    output_batch = []
    for source, answer, token_ids in zip(batch, answers, new_tokens):
        valid_tokens = int((token_ids != tokenizer.pad_token_id).sum().item())
        output_batch.append({
            "id": source["id"],
            "qid": source["qid"],
            "scenario": source["scenario"],
            "configuration": source["configuration"],
            "answer": answer.strip(),
            "references": source["references"],
            "generated_tokens": valid_tokens,
            "generation_seconds": elapsed / len(batch),
            "batch_generation_seconds": elapsed,
            "model_id": active_model_id,
            "model_revision": model_revision,
        })
    append_rows(OUTPUT_PATH, output_batch)
    for item in output_batch:
        cache[cache_key(item)] = item
    if start % (BATCH_SIZE * 10) == 0 or start + BATCH_SIZE >= len(pending):
        print(f"Progress {min(start + BATCH_SIZE, len(pending))}/{len(pending)} pending; cache={len(cache)}")

active_outputs = [cache[cache_key(row)] for row in selected]
assert len(active_outputs) == len(selected)
assert all(row["answer"] for row in active_outputs)
print("Generation complete:", len(active_outputs))
for item in active_outputs[:10]:
    print("\\n", item["id"], item["configuration"], "->", item["answer"][:500])'''
    ),
    code(
        '''# Free Qwen before loading the multilingual BERTScore encoder.
del model
gc.collect()
torch.cuda.empty_cache()

from bert_score import score as bert_score
from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def best_rouge(candidate, references):
    return max(rouge.score(reference, candidate)["rougeL"].fmeasure for reference in references)

candidates = [row["answer"] for row in active_outputs]
references = [row["references"] for row in active_outputs]
_, _, bert_f1 = bert_score(
    candidates,
    references,
    model_type="bert-base-multilingual-cased",
    batch_size=32,
    device="cuda",
    verbose=True,
)

abstain_text = "informasi tidak ditemukan dalam laporan"
metric_rows = []
for output, bert_value in zip(active_outputs, bert_f1.tolist()):
    metric_rows.append({
        "id": output["id"],
        "qid": output["qid"],
        "scenario": output["scenario"],
        "configuration": output["configuration"],
        "rouge_l_f1": best_rouge(output["answer"], output["references"]),
        "bertscore_f1": float(bert_value),
        "abstained": abstain_text in output["answer"].lower(),
    })

with METRICS_PATH.open("w", encoding="utf-8") as handle:
    for item in metric_rows:
        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\\n")
print("Metrics written:", METRICS_PATH, len(metric_rows))'''
    ),
    code(
        '''from collections import Counter
import hashlib
import numpy as np

expected = 25 if TEST_MODE else 1730
assert len(active_outputs) == expected
assert len(metric_rows) == expected
assert len({(r["id"], r["configuration"]) for r in active_outputs}) == expected
assert all(r["answer"].strip() for r in active_outputs)
assert all(np.isfinite(r["rouge_l_f1"]) and np.isfinite(r["bertscore_f1"]) for r in metric_rows)

counts = Counter((r["scenario"], r["configuration"]) for r in metric_rows)
print("Coverage:", dict(sorted(counts.items())))
print("Mean metrics by configuration:")
for config in sorted({r["configuration"] for r in metric_rows}):
    part = [r for r in metric_rows if r["configuration"] == config]
    print(config, {
        "rouge_l_f1": float(np.mean([r["rouge_l_f1"] for r in part])),
        "bertscore_f1": float(np.mean([r["bertscore_f1"] for r in part])),
        "abstain_rate": float(np.mean([r["abstained"] for r in part])),
    })

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("Output SHA256:", file_hash(OUTPUT_PATH))
print("Metrics SHA256:", file_hash(METRICS_PATH))
print("STAGE 11", "TEST PASS" if TEST_MODE else "FULL PASS")'''
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks" / "run_generation.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
