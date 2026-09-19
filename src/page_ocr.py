"""Read a FULL PAGE by segmenting it into lines, then recognising each line.

WHY THIS IS NEEDED: the trained model is a LINE recogniser -- it maps one
cropped text line to a character sequence. A full page is out of its domain, so
feeding a page in directly produces nonsense. Real OCR systems are two stages:

    page -> [detection/segmentation] -> line crops -> [recognition] -> text

This supplies the missing first stage. For clean documents (horizontal lines,
plain background) a neural detector is unnecessary -- a horizontal projection
profile finds the lines reliably and needs no training data.

METHOD
  1. Grayscale, then Otsu threshold -> binary (text = 1)
  2. Sum text pixels per row -> projection profile
  3. Smooth the profile, then take rows above a fraction of the mean as "text"
  4. Contiguous text rows form a line band; discard bands that are too thin
  5. Trim each band horizontally to its ink, pad, and recognise it

LIMITS (state these honestly in the report): assumes roughly horizontal,
non-overlapping lines and a fairly uniform background. Skewed photos, curved
lines, multi-column layouts and marginalia will break it.

    python -u src/page_ocr.py --image page.jpg --ckpt checkpoints/handwriting.pt
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from train_mac import CRNN, prep, pick_device

ROOT = Path(__file__).resolve().parent.parent


def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = w_b = 0.0
    best_t, best_var = 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def smooth(x, k=9):
    if k < 3:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def find_lines(img: Image.Image, min_height=12, thresh_frac=0.12, pad=4):
    g = np.array(img.convert("L"))
    t = otsu(g)
    ink = (g < t).astype(np.float32)          # 1 where text
    prof = smooth(ink.sum(axis=1), k=max(3, img.height // 200 * 2 + 1))
    cutoff = prof.mean() * thresh_frac
    is_text = prof > cutoff

    bands, start = [], None
    for i, v in enumerate(is_text):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_height:
                bands.append((start, i))
            start = None
    if start is not None and len(is_text) - start >= min_height:
        bands.append((start, len(is_text)))

    crops = []
    for y0, y1 in bands:
        y0 = max(0, y0 - pad); y1 = min(img.height, y1 + pad)
        strip = ink[y0:y1]
        cols = strip.sum(axis=0)
        nz = np.nonzero(cols > 0)[0]
        if len(nz) == 0:
            continue
        x0 = max(0, int(nz[0]) - pad); x1 = min(img.width, int(nz[-1]) + pad)
        if x1 - x0 < 16:
            continue
        crops.append((img.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)))
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "handwriting.pt"))
    ap.add_argument("--min-height", type=int, default=12)
    ap.add_argument("--thresh-frac", type=float, default=0.12)
    ap.add_argument("--out-vis", default=str(ROOT / "results" / "page_lines.png"))
    ap.add_argument("--out-txt", default=str(ROOT / "results" / "page_text.txt"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = pick_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    itos = {i + 1: c for i, c in enumerate(chars)}
    model = CRNN(len(chars) + 1).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"model: {a.ckpt}  ({len(chars)} classes, CER {ck.get('cer')})")

    img = Image.open(a.image)
    print(f"page:  {a.image}  {img.size[0]}x{img.size[1]}")
    crops = find_lines(img, a.min_height, a.thresh_frac)
    print(f"segmented into {len(crops)} lines\n")

    def decode(seq):
        o, p = [], None
        for i in seq:
            if i != 0 and i != p:
                o.append(itos.get(i, ""))
            p = i
        return "".join(o)

    vis = img.convert("RGB")
    d = ImageDraw.Draw(vis)
    lines = []
    with torch.no_grad():
        for n, (crop, box) in enumerate(crops, 1):
            x = prep(crop)
            arr = (np.array(x, dtype=np.float32) / 255.0 - 0.5) / 0.5
            lg = model(torch.from_numpy(arr)[None, None].to(dev))
            probs = lg.softmax(-1)[0]
            ids = lg.argmax(-1)[0].cpu().tolist()
            text = decode(ids)
            emitted = [i for i, t in enumerate(ids) if t != 0]
            conf = float(probs.max(-1).values[emitted].mean()) if emitted else 0.0
            lines.append(text)
            d.rectangle(box, outline=(220, 40, 40), width=3)
            print(f"{n:3d}. [{conf:.2f}] {text}")

    Path(a.out_vis).parent.mkdir(parents=True, exist_ok=True)
    vis.save(a.out_vis)
    Path(a.out_txt).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nline boxes -> {a.out_vis}")
    print(f"text       -> {a.out_txt}")
    print("\nCheck the boxes first: if segmentation is wrong the text will be too.")


if __name__ == "__main__":
    main()
