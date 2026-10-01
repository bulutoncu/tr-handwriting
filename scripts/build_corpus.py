"""Build clean Turkish sentence and line files, split by document.

From Turkish Wikipedia (needs `pip install -e ".[corpus]"`; streams, no full download):
    python scripts/build_corpus.py --out data/corpus --max-docs 50000

From your own text files (each blank-line-separated block counts as one document):
    python scripts/build_corpus.py --input my_texts/*.txt --out data/corpus
"""
import argparse
import random
from pathlib import Path

from tqdm import tqdm

from trhw.corpus import chunk_into_lines, good_sentence, split_for, split_sentences


def wikipedia_docs(max_docs: int):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        yield str(row["id"]), row["text"]


def file_docs(paths: list[str]):
    for p in paths:
        blocks = Path(p).read_text(encoding="utf-8").split("\n\n")
        for k, block in enumerate(blocks):
            yield f"{p}#{k}", block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", help="local UTF-8 text files instead of Wikipedia")
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--max-docs", type=int, default=50_000)
    ap.add_argument("--val-share", type=float, default=0.02)
    ap.add_argument("--test-share", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    files = {(kind, split): open(out / f"{kind}_{split}.txt", "w", encoding="utf-8")
             for kind in ("sentences", "lines") for split in ("train", "val", "test")}
    seen: set[str] = set()
    counts = {s: 0 for s in ("train", "val", "test")}

    docs = file_docs(args.input) if args.input else wikipedia_docs(args.max_docs)
    for doc_id, text in tqdm(docs, desc="documents"):
        split = split_for(doc_id, args.val_share, args.test_share)
        for sent in split_sentences(text):
            if not good_sentence(sent) or sent in seen:
                continue
            seen.add(sent)
            counts[split] += 1
            files["sentences", split].write(sent + "\n")
            for line in chunk_into_lines(sent, rng):
                files["lines", split].write(line + "\n")
    for f in files.values():
        f.close()
    print("sentences per split:", counts)


if __name__ == "__main__":
    main()
