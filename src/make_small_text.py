"""Generate small/low-resolution variants of the training data.

Two modes, for the two stages of the detect-then-read pipeline:

  recognizer  degrade the line crop in place -- shrink to a small pixel
              height, blur, JPEG, then restore the original size. Geometry is
              unchanged, detail is gone. This is what a photo of small print
              actually loses.

  detector    paste the line into a large white canvas at a random position
              and record the bounding box. Only useful with the box; the
              recognizer must never see these, because prep() rescales by
              HEIGHT and would shrink 24px text to 6px.

    python -u src/make_small_text.py --mode recognizer --level small
    python -u src/make_small_text.py --mode detector  --level very_small
"""
import argparse, io, json, random, sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from seeds import set_all_seeds, SEED

ROOT = Path(__file__).resolve().parent.parent

LEVELS = {
    "small":      [24, 28, 32, 36],
    "very_small": [12, 14, 16, 18, 20],
    "mixed":      [12, 14, 16, 18, 20, 24, 28, 32, 36],
}
CANVAS = {"small": (1024, 256), "very_small": (1536, 384), "mixed": (1280, 320)}


def degrade(img, target_h, rng, blur_p=0.30, jpeg=True):
    """Shrink to target_h, optionally blur, then restore the original size."""
    w, h = img.size
    small_w = max(4, int(w * target_h / max(h, 1)))
    out = img.resize((small_w, target_h), Image.Resampling.BILINEAR)
    if rng.random() < blur_p:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    out = out.resize((w, h), Image.Resampling.BICUBIC)
    if jpeg:
        buf = io.BytesIO()
        out.convert("L").save(buf, "JPEG", quality=rng.randint(40, 80))
        out = Image.open(buf)
    return out


def on_canvas(img, target_h, canvas_size, rng):
    """Paste a shrunken line at a random spot in a big white page."""
    w, h = img.size
    nw = max(4, int(w * target_h / max(h, 1)))
    cw, ch = canvas_size
    if nw >= cw - 20:
        nw = cw - 40
        target_h = max(4, int(h * nw / max(w, 1)))
    small = img.resize((nw, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", canvas_size, 255)
    x = rng.randint(10, max(10, cw - small.width - 10))
    y = rng.randint(10, max(10, ch - small.height - 10))
    canvas.paste(small, (x, y))
    if rng.random() < 0.25:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
    return canvas, (x, y, x + small.width, y + small.height)


def jpeg_bytes(img, q):
    buf = io.BytesIO()
    img.convert("L").save(buf, "JPEG", quality=q)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results" / "train_subset.pt"))
    ap.add_argument("--mode", choices=["recognizer", "detector"], default="recognizer")
    ap.add_argument("--level", choices=list(LEVELS), default="mixed")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    set_all_seeds()
    rng = random.Random(SEED)
    heights = LEVELS[a.level]

    bundle = torch.load(a.src, map_location="cpu", weights_only=False)
    pairs = bundle["train"] if "train" in bundle else \
            [(it["jpeg"], it["text"]) for it in bundle["items"]]
    if a.limit:
        pairs = pairs[:a.limit]
    print(f"{a.src}: {len(pairs)} samples -> mode={a.mode} level={a.level} "
          f"heights={heights}")

    out_items, boxes = [], []
    for i, (jb, text) in enumerate(pairs):
        img = Image.open(io.BytesIO(jb)).convert("L")
        th = rng.choice(heights)
        if a.mode == "recognizer":
            out_items.append((jpeg_bytes(degrade(img, th, rng), 90), text))
        else:
            canvas, box = on_canvas(img, th, CANVAS[a.level], rng)
            out_items.append((jpeg_bytes(canvas, rng.randint(75, 95)), text))
            boxes.append({"i": i, "text": text, "x0": box[0], "y0": box[1],
                          "x1": box[2], "y1": box[3],
                          "cw": CANVAS[a.level][0], "ch": CANVAS[a.level][1]})
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    out = Path(a.out) if a.out else \
        ROOT / "results" / f"train_{a.mode}_{a.level}.pt"
    payload = {"train": out_items, "chars": bundle.get("chars"),
               "seed": SEED, "mode": a.mode, "level": a.level,
               "heights": heights, "source": a.src}
    if a.mode == "detector":
        payload["boxes"] = boxes
        (out.with_suffix(".boxes.json")).write_text(json.dumps(boxes))
        print("wrote", out.with_suffix(".boxes.json"))
    torch.save(payload, out)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(out_items)} samples, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
