"""Evaluate one or more recognizers, alone and end to end with the mark restorer.

    python scripts/eval_recognizer.py --data data/synth/test \
        --models microsoft/trocr-small-handwritten checkpoints/synth-small/best

Columns:
  base CER        recognizer output vs base-shape label (what it was trained for)
  base CER (aA)   same, ignoring upper/lower case
  full, no marks  recognizer output vs real Turkish text (marks missing)
  full + restorer recognizer output -> hybrid restorer -> vs real Turkish text
Predictions are saved next to the data for error analysis.
"""
import argparse
import json
import re
from pathlib import Path

from trhw.metrics import cer, wer
from trhw.recognizer import load_image, load_model, pick_device, predict, read_labels
from trhw.restore import HybridRestorer, Restorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--restorers", default="models", help="folder with lexicon/context .pkl.gz")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--beams", type=int, default=1)
    args = ap.parse_args()

    rows = read_labels(args.data, args.n)
    images = [load_image(args.data, r) for r in rows]
    base = [r["base"] for r in rows]
    full = [r["text"] for r in rows]
    restorer = HybridRestorer(Restorer.load(Path(args.restorers) / "lexicon.pkl.gz"),
                              Restorer.load(Path(args.restorers) / "context.pkl.gz"))
    device = pick_device()

    table = []
    for name in args.models:
        print(f"evaluating {name} on {len(rows)} lines ({device})...")
        model, processor = load_model(name, device)
        preds = predict(model, processor, images, args.batch_size, num_beams=args.beams)
        restored = restorer.restore_many(preds)
        table.append((name, cer(base, preds), cer([b.lower() for b in base], [p.lower() for p in preds]),
                      cer(full, preds), cer(full, restored), wer(full, restored)))
        safe = re.sub(r"[^\w.-]+", "_", name)
        with open(Path(args.data) / f"predictions_{safe}.jsonl", "w", encoding="utf-8") as f:
            for r, p, rs in zip(rows, preds, restored):
                f.write(json.dumps({"image": r["image"], "text": r["text"], "base": r["base"],
                                    "pred": p, "restored": rs}, ensure_ascii=False) + "\n")
        del model

    print("\n| model | base CER | base CER (aA) | full, no marks | full + restorer | WER + restorer |")
    print("|---|---|---|---|---|---|")
    for name, b, bi, f, fr, w in table:
        print(f"| {name} | {b:.2%} | {bi:.2%} | {f:.2%} | {fr:.2%} | {w:.2%} |")


if __name__ == "__main__":
    main()
