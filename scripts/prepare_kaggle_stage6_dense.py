"""Prepare private Kaggle input for Stage 6 direct full-text query embeddings."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage6_selection import Resources, build_features, expanded_text


def main() -> None:
    processed = Path("data/processed")
    embeddings = Path("models/embeddings")
    output = Path("kaggle_io/stage6_dense")
    output.mkdir(parents=True, exist_ok=True)
    resources = Resources(processed, embeddings)
    features = build_features(resources, processed, [1, 2, 3], "QS-BM25")
    texts = [expanded_text(feature, source, 5, 3) for feature in features for source in ("q", "ext")]
    hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    unique = list(dict.fromkeys(zip(hashes, texts)))
    input_path = output / "stage6_dense_input.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for digest, text in unique:
            handle.write(json.dumps({"sha256": digest, "text": text}, ensure_ascii=False) + "\n")
    (output / "stage6_dense_sequence.json").write_text(
        json.dumps(hashes, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    manifest = {
        "rows": len(texts), "unique_rows": len(unique), "model_id": "intfloat/multilingual-e5-base",
        "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    archive = output / "stage6-dense-private.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(input_path, input_path.name)
        bundle.write(output / "manifest.json", "manifest.json")
    print(json.dumps({**manifest, "archive": str(archive), "archive_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
