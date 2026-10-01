"""Fine-tune TrOCR to read base-shape Turkish lines.

    python scripts/train_recognizer.py --train data/synth/train --val data/synth/val \
        --out checkpoints/synth-small --max-train 20000

Watch progress in the terminal; history.csv in the output folder has the
learning curve (loss and validation CER over time) for the README.
"""
import argparse
import json

from trhw.recognizer import LineDataset, TrainConfig, load_model, pick_device, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/trocr-small-handwritten",
                    help="Hugging Face model id or a local checkpoint folder")
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N training lines")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    model, processor = load_model(args.model, device)
    train_ds = LineDataset(args.train, processor, limit=args.max_train)
    print(f"{len(train_ds):,} training lines")
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
                      lr=args.lr, eval_every=args.eval_every, eval_n=args.eval_n,
                      num_workers=args.num_workers)
    result = train(model, processor, train_ds, args.val, args.out, cfg)
    print(json.dumps(result, indent=2))
    print(f"best checkpoint: {args.out}/best")


if __name__ == "__main__":
    main()
