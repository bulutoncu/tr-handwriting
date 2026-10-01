"""Render synthetic handwriting lines.

    python scripts/generate_synth.py --lines data/corpus/lines_train.txt \
        --out data/synth/train --n 100000

Writes images/*.png, labels.jsonl (text, base, font, marks_visible) and preview.png.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from trhw.fonts import load_manifest
from trhw.synth import make_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fonts", default="fonts")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turkish-fonts-only", action="store_true",
                    help="only fonts that draw the marks (for the visual mark-check experiments)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts = load_manifest(args.fonts)
    if args.turkish_fonts_only:
        fonts = [f for f in fonts if f["turkish"]]
    lines = [ln.strip() for ln in open(args.lines, encoding="utf-8") if ln.strip()]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    preview = []
    with open(out / "labels.jsonl", "w", encoding="utf-8") as labels:
        for i in tqdm(range(args.n), desc="rendering"):
            img, meta = make_sample(rng.choice(lines), fonts, rng)
            name = f"images/{i:07d}.png"
            cv2.imwrite(str(out / name), img)
            labels.write(json.dumps({"image": name, **meta}, ensure_ascii=False) + "\n")
            if len(preview) < 12:
                preview.append(img)
    width = max(p.shape[1] for p in preview)
    grid = np.vstack([np.pad(p, ((4, 4), (0, width - p.shape[1])), constant_values=255)
                      for p in preview])
    cv2.imwrite(str(out / "preview.png"), grid)
    print(f"wrote {args.n} samples to {out}/")


if __name__ == "__main__":
    main()
