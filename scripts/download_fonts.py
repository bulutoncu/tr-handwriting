"""Download handwriting fonts and record which ones fully support Turkish.

    python scripts/download_fonts.py --out fonts
"""
import argparse

from trhw.fonts import download_fonts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fonts")
    args = ap.parse_args()
    manifest = download_fonts(args.out)
    n_tr = sum(f["turkish"] for f in manifest)
    print(f"{len(manifest)} fonts ready ({n_tr} with full Turkish coverage) in {args.out}/")
