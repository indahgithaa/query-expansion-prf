from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_kaggle_embeddings import notebook


def test_embedding_notebook_contains_required_guards_and_outputs():
    payload = notebook()
    source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    assert "nvidia-smi" in source
    assert "INTERNET_OK" in source
    assert "intfloat/multilingual-e5-base" in source
    assert "query:" not in source  # prefixes are constructed from explicit kind at runtime
    assert "normalize_embeddings=True" in source
    assert "e5_base_embeddings" in source
    assert payload["metadata"]["kaggle"]["isInternetEnabled"] is True


def test_tracked_notebook_is_valid_when_present():
    path = Path("notebooks/run_embeddings.ipynb")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert len(payload["cells"]) >= 7
