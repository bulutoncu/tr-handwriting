"""Train the text-only mark restorers.

    python scripts/train_restorer.py --train data/corpus/sentences_train.txt --out models
"""
import argparse
from itertools import islice
from pathlib import Path

from trhw.restore import ContextRestorer, HybridRestorer, LexiconRestorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--max-sentences", type=int, default=300_000,
                    help="the context model's tables grow with data; raise this if you have the RAM")
    args = ap.parse_args()

    with open(args.train, encoding="utf-8") as f:
        sentences = [s.strip() for s in islice(f, args.max_sentences) if s.strip()]
    print(f"training on {len(sentences):,} sentences")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lexicon = LexiconRestorer().fit(sentences)
    print(f"lexicon: {len(lexicon.table):,} word forms")
    context = ContextRestorer().fit(sentences)
    print(f"context model: {len(context.stats):,} patterns")
    lexicon.save(out / "lexicon.pkl.gz")
    context.save(out / "context.pkl.gz")
    HybridRestorer(lexicon, context).save(out / "hybrid.pkl.gz")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
