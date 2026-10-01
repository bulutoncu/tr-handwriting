"""Turkish-aware text utilities.

Core idea of the project: the recognizer reads *base shapes* only, and a
separate model restores the Turkish marks. This module defines that mapping.

    ç→c  ğ→g  ı→i  ö→o  ş→s  ü→u      (and the uppercase versions, İ→I)

Note on i/ı: in the lowercase domain the base letter is "i" and the two
choices are dotted "i" and dotless "ı". Case is re-applied afterwards with
Turkish rules (i→İ, ı→I), so the uppercase I/İ ambiguity is handled for free.
"""

from __future__ import annotations

import re
import unicodedata

# lowercase base letter -> its "marked" alternative
LOWER_MARKED = {"c": "ç", "g": "ğ", "i": "ı", "o": "ö", "s": "ş", "u": "ü"}
AMBIGUOUS_LOWER = frozenset(LOWER_MARKED)

_TO_BASE = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
})

TURKISH_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
# q, w, x appear in loanwords, names and abbreviations, so we keep them.
EXTRA_LETTERS = "qwxQWX"
DIGITS = "0123456789"
PUNCT = " .,;:!?'\"()-/%&"
ALLOWED_CHARS = frozenset(TURKISH_LETTERS + EXTRA_LETTERS + DIGITS + PUNCT)
TURKISH_SPECIAL = "çğıöşüÇĞİÖŞÜ"

_CLEAN_MAP = str.maketrans({
    # circumflex vowels are optional in modern Turkish; fold them to keep scope small
    "â": "a", "î": "i", "û": "u", "Â": "A", "Î": "İ", "Û": "U",
    # typographic punctuation -> ASCII
    "’": "'", "‘": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‒": "-", "−": "-", "\u00a0": " ", "…": "...",
})


def to_base(text: str) -> str:
    """Strip Turkish marks: 'Çiğdem Işık' -> 'Cigdem Isik'. Length-preserving."""
    return text.translate(_TO_BASE)


def tr_lower(text: str) -> str:
    """Turkish lowercase (I→ı, İ→i). Length-preserving, unlike str.lower('İ')."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(text: str) -> str:
    """Turkish uppercase (i→İ, ı→I)."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def apply_case(char: str, like: str) -> str:
    """Give lowercase `char` the case of `like` using Turkish rules."""
    return tr_upper(char) if like.isupper() else char


def clean_text(text: str) -> str:
    """NFC-normalize, fold circumflexes and fancy punctuation, collapse spaces."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("i\u0307", "i")  # artefact of naive 'İ'.lower()
    text = text.translate(_CLEAN_MAP)
    return re.sub(r"\s+", " ", text).strip()


def is_allowed(text: str) -> bool:
    return bool(text) and all(ch in ALLOWED_CHARS for ch in text)


def has_turkish_marks(text: str) -> bool:
    return any(ch in TURKISH_SPECIAL for ch in text)


WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
