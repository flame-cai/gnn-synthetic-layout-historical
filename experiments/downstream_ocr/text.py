from __future__ import annotations

import re
import unicodedata

try:  # pragma: no cover - exercised when RapidFuzz is installed.
    from rapidfuzz.distance import Levenshtein as _RapidFuzzLevenshtein
except Exception:  # pragma: no cover - deterministic fallback is tested.
    _RapidFuzzLevenshtein = None


def normalize_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def levenshtein_distance(left: str, right: str) -> int:
    left = left or ""
    right = right or ""
    if _RapidFuzzLevenshtein is not None:
        return int(_RapidFuzzLevenshtein.distance(left, right))
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertions = previous[right_index] + 1
            deletions = current[right_index - 1] + 1
            substitutions = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]

