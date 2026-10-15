"""Prepare a private Kaggle input archive and executable embedding notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def notebook() -> dict:
    cells = [
        markdown_cell(
            """# Private multilingual E5 embedding cache

Notebook ini hanya untuk dataset Kaggle **Private**. Jangan mengubah visibilitas notebook atau dataset menjadi publik. Output tidak memuat teks klinis, hanya embedding, ID, dan manifest."""
        ),
        code_cell(
            """import os, platform, socket, subprocess, sys
print('python', sys.version)
print('platform', platform.platform())
subprocess.run(['nvidia-smi'], check=True)
try:
    socket.create_connection(('huggingface.co', 443), timeout=10).close()
    print('INTERNET_OK')
except Exception as exc:
    raise RuntimeError('Internet Kaggle harus On') from exc"""
        ),
        code_cell(
            """%pip install -q --upgrade sentence-transformers==6.1.0 huggingface-hub==1.33.0"""
        ),
        code_cell(
            """import hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

INPUT_ROOT = Path('/kaggle/input')
WORK = Path('/kaggle/working/prf_embeddings')
WORK.mkdir(parents=True, exist_ok=True)

def find_one(filename):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f'Diharapkan tepat satu {filename}, ditemukan {matches}')
    return matches[0]

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

chunks = read_jsonl(find_one('chunks.jsonl'))
qa = read_jsonl(find_one('qa.jsonl'))
print({'chunks': len(chunks), 'qa': len(qa), 'gpu': torch.cuda.get_device_name(0)})
assert len(chunks) == 13737 and len(qa) == 858"""
        ),
        code_cell(
            """MODEL_ID = 'intfloat/multilingual-e5-base'
REVISION = model_info(MODEL_ID).sha
model = SentenceTransformer(MODEL_ID, revision=REVISION, device='cuda')
model.max_seq_length = 512
dimension = int(model.get_sentence_embedding_dimension())
print({'model': MODEL_ID, 'revision': REVISION, 'dimension': dimension})

def encode_unique(texts, prefix, batch_size=64):
    unique = list(dict.fromkeys(texts))
    formatted = [f'{prefix}: {" ".join(text.split())}' for text in unique]
    vectors = model.encode(
        formatted, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    positions = {text: i for i, text in enumerate(unique)}
    expanded = np.asarray([vectors[positions[text]] for text in texts], dtype=np.float32)
    return expanded, len(unique)

started = time.perf_counter()
chunk_vectors, unique_chunks = encode_unique([r['text'] for r in chunks], 'passage')
question_vectors, unique_questions = encode_unique([r['question'] for r in qa], 'query')
answer_vectors, unique_answers = encode_unique([' '.join(r['answer_texts']) for r in qa], 'passage')
elapsed = time.perf_counter() - started
print({'elapsed_seconds': elapsed, 'unique': [unique_chunks, unique_questions, unique_answers]})"""
        ),
        code_cell(
            """for name, values, expected_rows in [
    ('chunks', chunk_vectors, len(chunks)),
    ('questions', question_vectors, len(qa)),
    ('answers', answer_vectors, len(qa)),
]:
    assert values.shape == (expected_rows, dimension), (name, values.shape)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5, rtol=0)

OUT = WORK / 'output'
OUT.mkdir(exist_ok=True)
np.save(OUT / 'chunks.npy', chunk_vectors)
np.save(OUT / 'questions.npy', question_vectors)
np.save(OUT / 'answers.npy', answer_vectors)
(OUT / 'ids.json').write_text(json.dumps({
    'chunk_ids': [r['chunk_id'] for r in chunks],
    'qids': [r['qid'] for r in qa],
}, ensure_ascii=False), encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest = {
    'model_id': MODEL_ID,
    'revision': REVISION,
    'device': torch.cuda.get_device_name(0),
    'elapsed_seconds': elapsed,
    'dimension': dimension,
    'rows': {'chunks': len(chunks), 'questions': len(qa), 'answers': len(qa)},
    'unique_texts': {'chunks': unique_chunks, 'questions': unique_questions, 'answers': unique_answers},
    'normalized': True,
}
for filename in ['chunks.npy', 'questions.npy', 'answers.npy', 'ids.json']:
    manifest.setdefault('sha256', {})[filename] = digest(OUT / filename)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))"""
        ),
        code_cell(
            """import shutil
archive = shutil.make_archive('/kaggle/working/e5_base_embeddings', 'zip', OUT)
print('OUTPUT_ARCHIVE', archive, Path(archive).stat().st_size)"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--io-dir", type=Path, default=Path("kaggle_io/embeddings"))
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/run_embeddings.ipynb"))
    args = parser.parse_args()

    required = [args.processed / "chunks.jsonl", args.processed / "qa.jsonl"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.io_dir.mkdir(parents=True, exist_ok=True)
    archive = args.io_dir / "embedding_inputs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in required:
            zf.write(path, path.name)

    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest = {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "members": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in required
        ],
        "notebook": str(args.notebook),
        "notebook_sha256": sha256(args.notebook),
        "privacy": {
            "required_dataset_visibility": "private",
            "required_notebook_visibility": "private",
            "delete_dataset_after_verified_download": True,
        },
    }
    (args.io_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
