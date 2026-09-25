"""Download a Khmer line corpus into images/ + labels.csv.

Writes exactly the layout generate_small_datasets.py reads:

    /content/existing_dataset/
        images/line_0000000.jpg
        labels.csv          filename,text

Streams from the Hub so nothing is held in RAM and you can stop at any N.

    python download_source_dataset.py --n 20000
"""
import argparse, io, os
from pathlib import Path

import pandas as pd
from PIL import Image
from datasets import load_dataset

def _base():
    """Kaggle writes to /kaggle/working, Colab to /content, otherwise cwd."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    if os.path.isdir("/content"):
        return "/content"
    return os.getcwd()


OUT = os.path.join(_base(), "existing_dataset")

# any of these may hold the transcript, depending on the corpus
TEXT_KEYS = ("text", "label", "transcription", "sentence", "caption", "ground_truth")


def get_text(row):
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def get_image(row):
    v = row.get("image") or row.get("img")
    if isinstance(v, Image.Image):
        return v
    if isinstance(v, dict) and "bytes" in v:
        return Image.open(io.BytesIO(v["bytes"]))
    if isinstance(v, (bytes, bytearray)):
        return Image.open(io.BytesIO(v))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Chanrith123333/khmer_english_ocr_image_line")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-chars", type=int, default=60,
                    help="skip longer lines; CTC needs timesteps >= 2x label")
    a = ap.parse_args()

    img_dir = Path(a.out) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"streaming {a.repo} [{a.split}] -> {a.out}")
    ds = load_dataset(a.repo, split=a.split, streaming=True)

    rows, kept, seen = [], 0, 0
    for row in ds:
        seen += 1
        text, img = get_text(row), get_image(row)
        if not text or img is None:
            continue
        if len(text) > a.max_chars:
            continue

        name = f"line_{kept:07d}.jpg"
        img.convert("L").save(img_dir / name, "JPEG", quality=92)
        rows.append({"filename": name, "text": text})
        kept += 1

        if kept % 1000 == 0:
            print(f"  {kept}/{a.n}  (scanned {seen})", flush=True)
        if kept >= a.n:
            break

    pd.DataFrame(rows).to_csv(Path(a.out) / "labels.csv", index=False)
    mb = sum(f.stat().st_size for f in img_dir.iterdir()) / 1e6
    print(f"\ndone: {kept} images, {mb:.0f} MB")
    print(f"  {a.out}/images/")
    print(f"  {a.out}/labels.csv")
    if rows:
        print(f"\nfirst label: {rows[0]['text']}")


if __name__ == "__main__":
    main()
