"""Multilingual E5 formatting, encoding, and normalized cache validation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def e5_text(text: str, *, kind: str) -> str:
    if kind not in {"query", "passage"}:
        raise ValueError("kind harus query atau passage")
    compact = " ".join(str(text).split())
    if not compact:
        raise ValueError("Teks embedding kosong")
    return f"{kind}: {compact}"


def validate_embeddings(values: np.ndarray, *, rows: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Matriks embedding tidak valid")
    if rows is not None and values.shape[0] != rows:
        raise ValueError("Jumlah baris embedding tidak cocok")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Embedding tidak ternormalisasi L2")
    return values


class E5Encoder:
    def __init__(self, model_id: str, *, cache_folder: Path, device: str = "cpu", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, cache_folder=str(cache_folder), device=device)
        self.model.max_seq_length = 512
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], *, kind: str, progress: bool = False) -> np.ndarray:
        formatted = [e5_text(text, kind=kind) for text in texts]
        values = self.model.encode(
            formatted,
            batch_size=self.batch_size,
            show_progress_bar=progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return validate_embeddings(values, rows=len(texts))


def encode_e5_transformers(
    texts: Sequence[str], *, model_id: str, revision: str, cache_folder: Path,
    kind: str, batch_size: int = 8, progress: bool = False,
) -> np.ndarray:
    """Encode E5 texts directly, avoiding sentence-transformers' optional SciPy import."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=str(cache_folder), local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_id, revision=revision, cache_dir=str(cache_folder), local_files_only=True,
    )
    model.eval()
    formatted = [e5_text(text, kind=kind) for text in texts]
    batches = range(0, len(formatted), batch_size)
    if progress:
        from tqdm import tqdm

        batches = tqdm(batches, total=(len(formatted) + batch_size - 1) // batch_size)
    outputs = []
    with torch.inference_mode():
        for start in batches:
            encoded = tokenizer(
                formatted[start:start + batch_size], max_length=512, padding=True,
                truncation=True, return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            outputs.append(functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    return validate_embeddings(np.concatenate(outputs).astype(np.float32), rows=len(texts))
