from pathlib import Path

import pytest

from scripts.verify_embeddings import verify


def test_downloaded_embedding_cache_when_present():
    path = Path("models/embeddings")
    if not path.exists():
        pytest.skip("Cache embedding tidak tersedia pada checkout ini")
    result = verify(Path("data/processed"), path)
    assert result["status"] == "PASS"
    assert result["normalized"] is True
