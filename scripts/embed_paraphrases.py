"""Encode the 96 Stage 4 paraphrases with the same pinned E5-base revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dense import e5_text, validate_embeddings


MODEL_ID = "intfloat/multilingual-e5-base"
REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/paraphrases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--cache", type=Path, default=Path("models/huggingface"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = [
        {"question_type": question_type, "partition": partition, "index": index, "text": text}
        for question_type, groups in payload.items()
        for partition in ("dev", "test")
        for index, text in enumerate(groups[partition])
    ]
    model = SentenceTransformer(MODEL_ID, revision=REVISION, cache_folder=str(args.cache), device="cpu")
    model.max_seq_length = 512
    started = time.perf_counter()
    values = model.encode(
        [e5_text(row["text"], kind="query") for row in records],
        batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    values = validate_embeddings(values, rows=96)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "paraphrases.npy", values)
    (args.output_dir / "paraphrase_ids.json").write_text(
        json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "model_id": MODEL_ID, "revision": REVISION, "kind": "query", "rows": 96,
        "dimension": int(values.shape[1]), "normalized": True,
        "elapsed_seconds": time.perf_counter() - started,
        "sha256": {
            name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
            for name in ("paraphrases.npy", "paraphrase_ids.json")
        },
    }
    (args.output_dir / "paraphrase_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
