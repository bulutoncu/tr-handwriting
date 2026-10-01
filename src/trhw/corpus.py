"""Turn raw Turkish text (e.g. Wikipedia articles) into clean sentences and
handwriting-length lines.

Splits are assigned per DOCUMENT, not per sentence: sentences from the same
article share names and phrasing, so splitting them across train/test would
leak and inflate your test scores.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterator

from .text import clean_text, is_allowed

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])")


def split_sentences(text: str) -> Iterator[str]:
    for paragraph in text.split("\n"):
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue
        yield from (s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip())


def good_sentence(s: str, min_words: int = 4, max_chars: int = 300) -> bool:
    if not is_allowed(s) or len(s) > max_chars or len(s.split()) < min_words:
        return False
    letters = sum(c.isalpha() for c in s)
    return letters / len(s) > 0.7  # drop tables, lists of numbers, references


def split_for(doc_id: str, val_share: float = 0.02, test_share: float = 0.02) -> str:
    h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test_share:
        return "test"
    if h < test_share + val_share:
        return "val"
    return "train"


def chunk_into_lines(sentence: str, rng: random.Random,
                     min_chars: int = 12, max_chars: int = 48) -> list[str]:
    """Pack words greedily into lines of random target length, like handwriting on paper."""
    lines, current = [], ""
    target = rng.randint(min_chars, max_chars)
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > target:
            lines.append(current)
            current = word
            target = rng.randint(min_chars, max_chars)
        else:
            current = candidate
    if current:
        lines.append(current)
    return [ln for ln in lines if len(ln) <= max_chars + 15]
