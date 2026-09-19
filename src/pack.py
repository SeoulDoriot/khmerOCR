"""Pack HF datasets into LMDB shards on LOCAL disk.

WHY NOT STRAIGHT TO DRIVE: mounted Google Drive costs ~10-50ms per file open.
At 1M images that is hours of pure I/O per epoch with the GPU idle. LMDB turns
the dataset into a handful of large files; Drive handles those fine, and we
copy them to /content at the start of each session.

Images are stored as JPEG bytes (already small, decode is fast). Labels are
normalized ONCE here, so training never re-normalizes.

Tiers for the curriculum are assigned by label length:
  char     1-2 chars
  word     3-12 chars
  sentence 13+ chars
"""
import argparse
import io
import json
from pathlib import Path

import lmdb
from datasets import load_dataset
from PIL import Image

from normalize import normalize_khmer

TEXT_KEYS = ["text", "label", "sentence", "transcription", "caption", "ground_truth"]
IMAGE_KEYS = ["image", "img", "picture"]
MAP_SIZE = 1 << 40   # 1TB virtual cap; LMDB only allocates what it uses


def tier_of(label):
    n = len(label)
    if n <= 2:
        return "char"
    if n <= 12:
        return "word"
    return "sentence"


def prep_image(img, height):
    """Grayscale, fixed height, aspect preserved, width clamped.

    Height 64 (not the Latin-OCR default of 32): Khmer stacks up to four
    vertical zones -- superscript vowel, base, subscript COENG, below-diacritic.
    At 32px the subscripts are destroyed and CER plateaus around 8-10%.
    """
    img = img.convert("L")
    w, h = img.size
    new_w = max(8, min(1024, int(w * height / max(h, 1))))
    return img.resize((new_w, height), Image.BILINEAR)


def find_key(row, candidates):
    for k in candidates:
        if k in row:
            return k
    return None


def pack(repo, out_dir, split="train", height=64, limit=None, quality=92):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_dir), map_size=MAP_SIZE, subdir=True)

    ds = load_dataset(repo, split=split, streaming=True)
    n, skipped = 0, 0
    counts = {"char": 0, "word": 0, "sentence": 0}
    txn = env.begin(write=True)

    for i, row in enumerate(ds):
        if limit and n >= limit:
            break
        tkey = find_key(row, TEXT_KEYS)
        ikey = find_key(row, IMAGE_KEYS)
        if tkey is None or ikey is None:
            skipped += 1
            continue
        label = normalize_khmer(row[tkey])
        if not label:
            skipped += 1
            continue
        try:
            img = prep_image(row[ikey], height)
        except Exception:
            skipped += 1
            continue

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        t = tier_of(label)
        counts[t] += 1

        txn.put(f"img-{n:09d}".encode(), buf.getvalue())
        txn.put(f"lbl-{n:09d}".encode(), label.encode())
        txn.put(f"tier-{n:09d}".encode(), t.encode())
        n += 1

        if n % 5000 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"  {n} packed ({skipped} skipped)", flush=True)

    txn.put(b"num-samples", str(n).encode())
    txn.put(b"meta", json.dumps(
        {"repo": repo, "height": height, "counts": counts}).encode())
    txn.commit()
    env.sync()
    env.close()
    print(f"done {repo}: {n} samples, {skipped} skipped, tiers={counts}")
    return n, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    pack(args.repo, args.out, args.split, args.height, args.limit)


if __name__ == "__main__":
    main()
