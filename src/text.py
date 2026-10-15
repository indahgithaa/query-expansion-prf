"""Shared text normalization used by retrieval and distant-supervision labels."""

from __future__ import annotations

import re
from functools import lru_cache

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
EXTRA_STOPWORDS = {"apa", "bagaimana", "yang", "ini", "pasien"}


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    return frozenset(StopWordRemoverFactory().get_stop_words()) | EXTRA_STOPWORDS


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    values = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        blocked = stopwords()
        return [token for token in values if token not in blocked]
    return values
