"""Restoring Turkish marks from context ("deasciification").

Input: base-shape text as produced by the recognizer, e.g. "Cocuk okula gitti".
Output: "Çocuk okula gitti".

Models, from simplest to strongest:
  IdentityRestorer  - never adds marks. The floor every model must beat.
  LexiconRestorer   - most frequent spelling of each known word. Fails on
                      unseen words, which agglutinative Turkish produces constantly.
  ContextRestorer   - decides each ambiguous letter from the surrounding
                      characters, backing off from wide to narrow windows.
                      Generalizes to unseen words.
  HybridRestorer    - lexicon for frequent words, context model otherwise.

Next step (see README roadmap): a neural character model, and combining
these text-only decisions with visual evidence from the image.
"""

from __future__ import annotations

import gzip
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .text import AMBIGUOUS_LOWER, LOWER_MARKED, WORD_RE, apply_case, to_base, tr_lower


class Restorer:
    def restore(self, base_text: str) -> str:
        raise NotImplementedError

    def restore_many(self, texts: Iterable[str]) -> list[str]:
        return [self.restore(t) for t in texts]

    def save(self, path: str | Path) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "Restorer":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


class IdentityRestorer(Restorer):
    def restore(self, base_text: str) -> str:
        return base_text


class LexiconRestorer(Restorer):
    def __init__(self, min_count: int = 1):
        self.min_count = min_count
        self.table: dict[str, str] = {}
        self.counts: dict[str, int] = {}

    def fit(self, texts: Iterable[str]) -> "LexiconRestorer":
        forms: dict[str, Counter] = defaultdict(Counter)
        for text in texts:
            for w in WORD_RE.findall(text):
                lw = tr_lower(w)
                forms[to_base(lw)][lw] += 1
        for key, counter in forms.items():
            total = sum(counter.values())
            if total >= self.min_count:
                self.table[key] = counter.most_common(1)[0][0]
                self.counts[key] = total
        return self

    @property
    def vocabulary(self) -> set[str]:
        return set(self.table)

    def restore_word(self, word: str) -> str | None:
        choice = self.table.get(word.lower())
        if choice is None:
            return None
        return "".join(apply_case(c, like) for c, like in zip(choice, word))

    def restore(self, base_text: str) -> str:
        out = list(base_text)
        for m in WORD_RE.finditer(base_text):
            restored = self.restore_word(m.group())
            if restored is not None:
                out[m.start():m.end()] = restored
        return "".join(out)


class ContextRestorer(Restorer):
    """Decision per ambiguous letter from (left, right) character windows of the
    base text, backing off from the widest window with enough evidence."""

    def __init__(self, windows=((5, 5), (4, 4), (3, 3), (4, 1), (1, 4), (2, 2), (1, 1), (0, 0)),
                 min_count: int = 2, min_margin: float = 0.6):
        self.windows = tuple(windows)
        self.min_count = min_count
        self.min_margin = min_margin
        self.stats: dict[str, list[int]] = {}

    @staticmethod
    def _pad(s: str, width: int) -> str:
        return "^" * width + s + "$" * width

    def _keys(self, padded: str, pos: int):
        for left, right in self.windows:
            yield f"{left},{right}:{padded[pos - left:pos + right + 1]}"

    def fit(self, texts: Iterable[str]) -> "ContextRestorer":
        stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        width = max(max(w) for w in self.windows)
        for text in texts:
            gold = tr_lower(text)
            base = to_base(gold)
            padded = self._pad(base, width)
            for k, ch in enumerate(base):
                if ch in AMBIGUOUS_LOWER:
                    label = int(gold[k] != ch)  # 1 = marked letter
                    for key in self._keys(padded, k + width):
                        stats[key][label] += 1
        self.stats = dict(stats)
        return self

    def decide(self, padded: str, pos: int) -> bool:
        for key in self._keys(padded, pos):
            counts = self.stats.get(key)
            if counts is None:
                continue
            total = counts[0] + counts[1]
            share = counts[1] / total
            if total >= self.min_count and max(share, 1 - share) >= self.min_margin:
                return share > 0.5
        return False

    def restore(self, base_text: str) -> str:
        width = max(max(w) for w in self.windows)
        lower = base_text.lower()
        padded = self._pad(lower, width)
        out = []
        for k, (ch, orig) in enumerate(zip(lower, base_text)):
            if ch in AMBIGUOUS_LOWER and self.decide(padded, k + width):
                out.append(apply_case(LOWER_MARKED[ch], orig))
            elif ch in AMBIGUOUS_LOWER:
                out.append(apply_case(ch, orig))  # handles I -> İ for dotted i
            else:
                out.append(orig)
        return "".join(out)


class HybridRestorer(Restorer):
    """Trust the lexicon for words seen often enough, the context model otherwise."""

    def __init__(self, lexicon: LexiconRestorer, context: ContextRestorer, min_word_count: int = 3):
        self.lexicon = lexicon
        self.context = context
        self.min_word_count = min_word_count

    def restore(self, base_text: str) -> str:
        out = list(self.context.restore(base_text))
        for m in WORD_RE.finditer(base_text):
            word = m.group()
            if self.lexicon.counts.get(word.lower(), 0) >= self.min_word_count:
                restored = self.lexicon.restore_word(word)
                if restored is not None:
                    out[m.start():m.end()] = restored
        return "".join(out)
