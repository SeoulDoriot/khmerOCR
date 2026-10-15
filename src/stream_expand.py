"""Stream the 12.1M corpus and write expanded shards until a disk budget is hit.

Shards so a crash costs one shard, not the whole run. Resumes by counting the
shards already written.

    python -u src/stream_expand.py --target 100000 --shard 25000
"""
import argparse, io, random, sys, time
from pathlib import Path

import torch
from PIL import Image
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from seeds import set_all_seeds, SEED
from expand_canvas import expand

ROOT = Path(__file__).resolve().parent.parent


def jbytes(img, q):
    b = io.BytesIO(); img.convert("L").save(b, "JPEG", quality=q); return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Chanrith123333/khmer_english_ocr_image_line")
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--shard", type=int, default=25000)
    ap.add_argument("--max-chars", type=int, default=60)
    ap.add_argument("--expansion", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--keep-clean", type=float, default=0.30)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "stream_expanded"))
    a = ap.parse_args()

    set_all_seeds()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    done = sorted(out_dir.glob("shard_*.pt"))
    skip = len(done) * a.shard
    print(f"{len(done)} shards present ({skip:,} images) -- resuming", flush=True)
    if skip >= a.target:
        print("target already reached"); return

    ds = load_dataset(a.repo, split="train", streaming=True)
    if skip:
        ds = ds.skip(skip)
    stream = iter(ds)          # ONE iterator for the whole run -- re-entering
                               # `for row in ds` would restart it from row 0
                               # and every shard would hold identical data.

    shard_i, total, t0 = len(done), skip, time.time()
    while total < a.target:
        rng = random.Random(SEED + shard_i)
        items, n = [], 0
        for row in stream:
            text = row.get("text")
            if not text or len(text) > a.max_chars:
                continue
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img["bytes"]))
            img = img.convert("L")
            if rng.random() < a.keep_clean:
                items.append((jbytes(img, 92), text))
            else:
                c, _, _ = expand(img, rng.choice(a.expansion), "portrait", rng)
                items.append((jbytes(c, rng.randint(75, 95)), text))
            n += 1; total += 1
            if n % 5000 == 0:
                r = (total - skip) / max(time.time() - t0, 1)
                print(f"  shard {shard_i}: {n}/{a.shard}  total {total:,}  {r:.0f} img/s", flush=True)
            if n >= a.shard or total >= a.target:
                break
        if not items:
            print("stream exhausted"); break
        p = out_dir / f"shard_{shard_i:03d}.pt"
        torch.save({"train": items, "seed": SEED, "shard": shard_i,
                    "expansion": a.expansion, "keep_clean": a.keep_clean}, p)
        print(f"wrote {p.name}  {len(items)} imgs  {p.stat().st_size/1e6:.0f} MB  "
              f"total {total:,}/{a.target:,}", flush=True)
        shard_i += 1

    print(f"done: {total:,} images in {shard_i} shards -> {out_dir}")


if __name__ == "__main__":
    main()
