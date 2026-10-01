"""Metrics.

- cer / wer: standard recognition metrics.
- restoration_report: how well marks are restored (input and output have equal length).
- correction_report: for any post-correction step (language model etc.), counts
  errors it FIXED versus correct characters it BROKE. The 'broken' number is
  the one that tells you whether the language model is hallucinating.
"""

from __future__ import annotations

from collections.abc import Sequence

from .text import AMBIGUOUS_LOWER, WORD_RE, to_base, tr_lower


def edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    errors = sum(edit_distance(r, h) for r, h in zip(refs, hyps, strict=True))
    return errors / max(1, sum(len(r) for r in refs))


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    errors = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps, strict=True))
    return errors / max(1, sum(len(r.split()) for r in refs))


def align_matches(ref: str, hyp: str) -> list[bool]:
    """For each character of `ref`, was it matched exactly in `hyp`? (Levenshtein backtrace)"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    matched = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            matched[i - 1] = True
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return matched


def correction_report(refs: Sequence[str], before: Sequence[str], after: Sequence[str]) -> dict:
    fixed = broken = still_wrong = total = 0
    for r, b, a in zip(refs, before, after, strict=True):
        mb, ma = align_matches(r, b), align_matches(r, a)
        total += len(r)
        for ok_b, ok_a in zip(mb, ma):
            fixed += (not ok_b) and ok_a
            broken += ok_b and (not ok_a)
            still_wrong += (not ok_b) and (not ok_a)
    return {"chars": total, "fixed": fixed, "broken": broken, "still_wrong": still_wrong,
            "net_gain": fixed - broken}


def restoration_report(refs: Sequence[str], preds: Sequence[str],
                       seen_words: set[str] | None = None) -> dict:
    """Mark-restoration quality. `refs` are gold Turkish texts, `preds` the restored
    versions of to_base(ref). If `seen_words` (lowercase base forms from training)
    is given, word accuracy is also reported separately for unseen words."""
    amb_total = amb_correct = 0
    words_total = words_correct = 0
    unseen_total = unseen_correct = 0
    for ref, pred in zip(refs, preds, strict=True):
        if len(ref) != len(pred):
            raise ValueError(f"length mismatch: {ref!r} vs {pred!r}")
        lref = tr_lower(ref)
        for k, ch in enumerate(lref):
            if to_base(ch) in AMBIGUOUS_LOWER:
                amb_total += 1
                amb_correct += ref[k] == pred[k]
        for match in WORD_RE.finditer(ref):
            w_ref, w_pred = match.group(), pred[match.start():match.end()]
            if not any(to_base(c) in AMBIGUOUS_LOWER for c in tr_lower(w_ref)):
                continue  # nothing to decide in this word
            words_total += 1
            ok = w_ref == w_pred
            words_correct += ok
            if seen_words is not None and to_base(w_ref).lower() not in seen_words:
                unseen_total += 1
                unseen_correct += ok
    out = {
        "ambiguous_char_acc": amb_correct / max(1, amb_total),
        "ambiguous_word_acc": words_correct / max(1, words_total),
        "ambiguous_words": words_total,
    }
    if seen_words is not None:
        out["unseen_word_acc"] = unseen_correct / max(1, unseen_total)
        out["unseen_word_share"] = unseen_total / max(1, words_total)
    return out
