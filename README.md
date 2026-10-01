# Turkish Handwriting Recognition: read the shapes, restore the marks

Turkish handwriting recognition is underserved: public line-level datasets are
almost nonexistent, and the hardest part of Turkish script is exactly what
general models get wrong — the small marks in **ç ğ ı İ ö ş ü**.

This project splits the problem in two:

```
photo ──► line detection ──► base-shape recognizer ──► mark restorer ──► text
                              "Cocuk okula gitti"      "Çocuk okula gitti"
                              (reads c g i o s u,       (decides ç/c, ş/s, ı/i …
                               ignores marks)            from context and meaning)
```

**Why split it?**

1. **Data.** A recognizer that ignores marks can learn from English handwriting
   (IAM) and from synthetic Turkish text, which removes most of the data problem.
2. **Context does the rest.** Humans read "cocuk" as "çocuk" without seeing the
   cedilla. A text model trained on Turkish can do the same, and the two
   models can be developed, measured and improved independently.
3. **Meaning matters.** Some words are ambiguous alone — *oldu / öldü*,
   *kaş / kas* — and only the sentence decides. That's where the error
   analysis gets interesting.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Text foundation: Turkish-aware normalization, metrics, corpus pipeline | ✅ |
| 2 | Mark restorer baselines: lexicon, context backoff model, hybrid | ✅ |
| 3 | Synthetic handwriting generator (29 fonts, augmentations, damage tools) | ✅ |
| 4 | Base-shape recognizer: fine-tune TrOCR on synthetic + IAM | ⏳ |
| 5 | Real handwriting test set (own notes + volunteers) | ⏳ |
| 6 | Neural restorer (character model / BERTurk) + combining with visual evidence | ⏳ |
| 7 | Robustness experiment: damaged images, with vs without context | ⏳ |
| 8 | Line segmentation + Gradio demo on Hugging Face Spaces | ⏳ |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[corpus,dev]"
python scripts/download_fonts.py                    # 29 handwriting fonts, 21 with full Turkish
python scripts/build_corpus.py --max-docs 20000     # Turkish Wikipedia, split by article
python scripts/train_restorer.py --train data/corpus/sentences_train.txt
python scripts/eval_restorer.py --test data/corpus/sentences_test.txt
python scripts/generate_synth.py --lines data/corpus/lines_train.txt --out data/synth/train --n 100000
python -m pytest
```

## Design decisions

- **Splits are per document, not per sentence.** Sentences from the same
  Wikipedia article share names and phrasing; splitting them across train and
  test would inflate scores.
- **Length-preserving text transforms.** `to_base`, `tr_lower` and `tr_upper`
  never change string length (Python's `"İ".lower()` does!), so restorer
  output can be compared to the reference letter by letter.
- **Turkish casing.** Restorers work in lowercase and decide *dotted i vs
  dotless ı*; case is re-applied with Turkish rules, so I/İ comes for free.
- **Circumflexes (â, î, û) are folded** to plain vowels to keep scope small.
- **Fonts missing Turkish glyphs still help:** they render the base-shape
  version of a line, which is all the recognizer needs. All-caps fonts are
  excluded because they would make the labels wrong.

## Metrics

- **CER / WER** — standard recognition metrics. CER is the headline number:
  Turkish is agglutinative, so WER punishes a single wrong suffix hard.
- **Ambiguous-letter accuracy** — accuracy on positions where a mark decision
  exists (c/ç, g/ğ, i/ı, o/ö, s/ş, u/ü).
- **Unseen-word accuracy** — the same, restricted to words never seen in
  training. This is where lexicon approaches collapse and context models earn
  their keep.
- **Fixed vs broken** (`correction_report`) — for any language-model
  correction step, how many errors it fixed *and how many correct letters it
  broke*. A language model that "corrects" names and rare words into common
  ones shows up here.

## Results

### Mark restoration (text only)

**Setup:** corpus built from 20,000 Turkish Wikipedia articles (743,636
training sentences after cleaning). Restorers trained on the first 100,000
training sentences and evaluated on 5,000 held-out sentences from articles
never seen in training. Input is the gold text with all marks removed; the
task is to put them back.

| Restorer | CER | Ambiguous letters | Ambiguous words | Unseen words |
|---|---|---|---|---|
| No marks (floor) | 8.72% | 65.85% | 48.54% | 54.61% |
| Lexicon | 0.62% | 97.57% | 96.25% | 54.61% |
| Context backoff | 0.61% | 97.61% | 95.04% | 79.34% |
| **Hybrid** | **0.38%** | **98.51%** | **96.99%** | **79.34%** |

Unseen words make up 4.7% of the ambiguous words in the test set.

**Findings**

- **The problem is real.** Leaving the marks out gives a CER of 8.72%: roughly
  one character in eleven needs a mark, and about half of the words that
  contain a mark decision come out wrong.
- **A lexicon cannot handle unseen words.** It is strong on words it knows
  (96.25%), but on unseen words it stays exactly at the floor (54.61%),
  because it leaves unknown words untouched. In an agglutinative language this
  limit is unavoidable.
- **The context model closes most of that gap.** On unseen words it goes from
  54.61% to 79.34%, because it learns the character patterns around each
  letter rather than whole words. Suffix patterns such as *-lük / -lik*
  transfer to words it has never seen.
- **The hybrid gets the best of both.** Trusting the lexicon for frequent
  words and the context model elsewhere brings CER from 8.72% down to 0.38%,
  removing about 96% of the errors.

**Caveats**

- **The test text is also Wikipedia, so unseen words are rare (4.7%).** Real
  handwritten notes come from a different domain — lecture notes, technical
  terms, abbreviations, names — where the unseen-word share will be much
  higher, and the context model will matter more. Measuring this on real
  pages is part of phase 5.
- **The input here is perfect unmarked text.** In the full system the input
  is the recognizer's imperfect reading: letter errors will also disturb the
  mark decisions, so errors compound. End-to-end evaluation comes in the
  later phases.

## Project layout

```
src/trhw/
  text.py      Turkish-aware normalization, base-shape mapping, casing
  metrics.py   CER/WER, alignment, restoration + correction reports
  corpus.py    sentence splitting, filtering, line chunking, leakage-safe splits
  restore.py   mark restorers (identity, lexicon, context, hybrid)
  fonts.py     font download + Turkish glyph coverage checks
  synth.py     synthetic handwriting renderer with augmentations
  damage.py    controlled damage (remove marks, erase patches) for robustness tests
scripts/       command-line entry points
tests/         pytest suite
```

## Fonts

Fonts are downloaded from the [Google Fonts repository](https://github.com/google/fonts)
and are OFL or Apache licensed. They are not committed to this repo.