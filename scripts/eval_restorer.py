"""Compare restorers on held-out sentences and print a Markdown table for the README.

    python scripts/eval_restorer.py --test data/corpus/sentences_test.txt --models models
"""
import argparse
from itertools import islice
from pathlib import Path

from trhw.metrics import cer, restoration_report
from trhw.restore import IdentityRestorer, Restorer
from trhw.text import to_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--max-sentences", type=int, default=5_000)
    args = ap.parse_args()

    with open(args.test, encoding="utf-8") as f:
        refs = [s.strip() for s in islice(f, args.max_sentences) if s.strip()]
    inputs = [to_base(r) for r in refs]
    models: dict[str, Restorer] = {"no marks (floor)": IdentityRestorer()}
    for name in ("lexicon", "context", "hybrid"):
        models[name] = Restorer.load(Path(args.models) / f"{name}.pkl.gz")
    seen = models["lexicon"].vocabulary

    print(f"{len(refs):,} test sentences\n")
    print("| model | CER | ambiguous letters | ambiguous words | unseen words |")
    print("|---|---|---|---|---|")
    report = {}
    for name, model in models.items():
        preds = model.restore_many(inputs)
        report = restoration_report(refs, preds, seen)
        print(f"| {name} | {cer(refs, preds):.2%} | {report['ambiguous_char_acc']:.2%} "
              f"| {report['ambiguous_word_acc']:.2%} | {report['unseen_word_acc']:.2%} |")
    print(f"\nunseen words are {report['unseen_word_share']:.1%} of ambiguous test words")


if __name__ == "__main__":
    main()
