"""Shrink text by growing the image around it, not by resizing the text.

The recognizer normalises every image to height 64. So if a line crop is padded
until it is N times taller than the original, the text arrives at the model at
64/N pixels:

    expansion 2.0x  ->  32px text   (easy)
    expansion 3.2x  ->  20px text   (starts hurting)
    expansion 4.0x  ->  16px text   (the measured cliff, 40% CER)
    expansion 5.3x  ->  12px text   (74% CER)

Padding also drives the aspect ratio down, which is the other thing that hurts:
samples at aspect 0-3 average 37.92% CER against 3.77% at aspect 10-16. A
portrait-shaped pad produces both failure modes at once, which is exactly the
case worth training on.

    python -u src/expand_canvas.py --expansion 4.0 --shape portrait
"""
import argparse, io, json, random, sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from seeds import set_all_seeds, SEED

ROOT = Path(__file__).resolve().parent.parent
PREP_H = 64  # must match train_mac.H


def expand(img, expansion, shape, rng, jitter=0.25, blur_p=0.25):
    """Pad around the text so it occupies 1/expansion of the height.

    shape: 'portrait'  taller than wide  (worst case: low aspect ratio)
           'balanced'  pad both axes equally
           'wide'      pad mostly horizontally (keeps aspect ratio high)
    """
    w, h = img.size
    new_h = max(h + 2, int(round(h * expansion)))

    if shape == "portrait":
        new_w = max(w + 2, int(round(new_h * rng.uniform(0.5, 0.9))))
        new_w = max(new_w, w + 2)
    elif shape == "wide":
        new_w = int(round(w * expansion))
    else:
        new_w = max(w + 2, int(round(w * expansion)))

    canvas = Image.new("L", (new_w, new_h), 255)

    # random placement, biased toward centre by `jitter`
    free_x, free_y = new_w - w, new_h - h
    cx, cy = free_x // 2, free_y // 2
    x = int(cx + rng.uniform(-jitter, jitter) * free_x)
    y = int(cy + rng.uniform(-jitter, jitter) * free_y)
    x = max(0, min(free_x, x))
    y = max(0, min(free_y, y))

    canvas.paste(img, (x, y))

    if rng.random() < blur_p:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.6)))

    box = (x, y, x + w, y + h)
    return canvas, box, PREP_H / (new_h / h)


def jpeg_bytes(img, q):
    buf = io.BytesIO()
    img.convert("L").save(buf, "JPEG", quality=q)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results" / "train_subset.pt"))
    ap.add_argument("--expansion", type=float, nargs="+", default=[2.0, 3.2, 4.0, 5.3],
                    help="canvas height multiplier(s); one is picked per sample")
    ap.add_argument("--shape", choices=["portrait", "balanced", "wide"],
                    default="portrait")
    ap.add_argument("--keep-clean", type=float, default=0.30,
                    help="fraction left untouched, so normal text is not forgotten")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    set_all_seeds()
    rng = random.Random(SEED)

    bundle = torch.load(a.src, map_location="cpu", weights_only=False)
    pairs = bundle["train"] if "train" in bundle else \
            [(it["jpeg"], it["text"]) for it in bundle["items"]]
    if a.limit:
        pairs = pairs[:a.limit]

    print(f"{len(pairs)} samples · shape={a.shape} · expansion={a.expansion} · "
          f"keep_clean={a.keep_clean:.0%}")
    print("effective text height after prep():")
    for e in a.expansion:
        print(f"   {e:4.1f}x  ->  {PREP_H/e:5.1f}px")
    print()

    out_items, meta, clean = [], [], 0
    for i, (jb, text) in enumerate(pairs):
        img = Image.open(io.BytesIO(jb)).convert("L")
        if rng.random() < a.keep_clean:
            out_items.append((jpeg_bytes(img, 92), text))
            clean += 1
            continue
        e = rng.choice(a.expansion)
        canvas, box, eff = expand(img, e, a.shape, rng)
        out_items.append((jpeg_bytes(canvas, rng.randint(75, 95)), text))
        meta.append({"i": i, "expansion": e, "eff_text_px": round(eff, 1),
                     "canvas": list(canvas.size), "box": list(box),
                     "aspect": round(canvas.size[0] / canvas.size[1], 2)})
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(pairs)}", flush=True)

    out = Path(a.out) if a.out else ROOT / "results" / f"train_expanded_{a.shape}.pt"
    torch.save({"train": out_items, "chars": bundle.get("chars"), "seed": SEED,
                "shape": a.shape, "expansion": a.expansion,
                "keep_clean": a.keep_clean, "meta": meta, "source": a.src}, out)

    if meta:
        ars = [m["aspect"] for m in meta]
        print(f"\n{len(out_items)} samples ({clean} left clean)")
        print(f"aspect ratio: min {min(ars):.2f}  median "
              f"{sorted(ars)[len(ars)//2]:.2f}  max {max(ars):.2f}")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
