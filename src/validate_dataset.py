"""Check a generated dataset is actually usable before spending GPU hours on it.

Six checks, in increasing order of how much they tell you:

  1 structure    every sample decodes, has a label, image is non-trivial
  2 charset      no label contains a character the model cannot emit
  3 length       no label exceeds the CTC ceiling (2*len <= W//4)
  4 content      the image is not blank -- catches silently empty canvases
  5 boxes        every box lies inside its canvas and is non-degenerate
  6 round-trip   crop by the saved box, run the recognizer, compare to the
                 label. This is the only check that proves the boxes point at
                 the text rather than at empty paper.

    python -u src/validate_dataset.py --set results/dataset_10_detector.pt
"""
import argparse, io, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_mac import CRNN, prep, edit, pick_device

ROOT = Path(__file__).resolve().parent.parent
W, H, MAX_LEN = 512, 64, 60


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    m = CRNN(len(chars) + 1).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, chars, {i + 1: c for i, c in enumerate(chars)}


@torch.no_grad()
def read(model, itos, imgs, dev, bs=48):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([
            torch.from_numpy((np.array(prep(im), dtype=np.float32) / 255. - .5) / .5)[None]
            for im in imgs[i:i + bs]]).to(dev)
        for s in model(x).argmax(-1).cpu().tolist():
            o, p = [], None
            for t in s:
                if t != 0 and t != p:
                    o.append(itos.get(t, ""))
                p = t
            out.append("".join(o))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--ckpt", default="checkpoints/A2_crnn_transfer.pt")
    ap.add_argument("--roundtrip", type=int, default=200)
    a = ap.parse_args()

    b = torch.load(a.set, map_location="cpu", weights_only=False)
    items = b["train"] if "train" in b else b["items"]
    meta = {m["i"]: m for m in b.get("meta", [])}
    chars = set(b.get("chars") or [])
    print(f"{a.set}\n{len(items)} samples · charset {len(chars)} · "
          f"{len(meta)} with boxes\n")

    fails = Counter()
    sizes, blanks, lens = [], 0, []
    bad_char = Counter()

    for i, (jb, text) in enumerate(items):
        try:
            im = Image.open(io.BytesIO(jb)).convert("L")
        except Exception:
            fails["undecodable"] += 1
            continue
        sizes.append(im.size)

        if not text or not text.strip():
            fails["empty_label"] += 1
        lens.append(len(text))
        if len(text) > MAX_LEN:
            fails["label_too_long"] += 1
        for c in set(text) - chars:
            bad_char[c] += 1

        arr = np.asarray(im, dtype=np.uint8)
        if arr.std() < 2.0:
            blanks += 1

        m = meta.get(i)
        if m:
            x0, y0, x1, y1 = m["box"]
            cw, ch = m["canvas"]
            if not (0 <= x0 < x1 <= cw and 0 <= y0 < y1 <= ch):
                fails["box_outside_canvas"] += 1
            if (x1 - x0) < 4 or (y1 - y0) < 4:
                fails["box_degenerate"] += 1

    print("1 structure   ", "OK" if not fails["undecodable"] else f"{fails['undecodable']} undecodable")
    print("2 charset     ", "OK" if not bad_char else
          f"{sum(bad_char.values())} chars outside charset: {list(bad_char)[:8]}")
    print(f"3 length      ", "OK" if not fails["label_too_long"] else
          f"{fails['label_too_long']} labels over {MAX_LEN}")
    print(f"              min {min(lens)} / mean {sum(lens)/len(lens):.0f} / max {max(lens)}")
    print("4 content     ", "OK" if not blanks else f"{blanks} near-blank images")
    if meta:
        bad = fails["box_outside_canvas"] + fails["box_degenerate"]
        print("5 boxes       ", "OK" if not bad else f"{bad} invalid")
    else:
        print("5 boxes        (none saved)")

    ws = [s[0] for s in sizes]; hs = [s[1] for s in sizes]
    ars = sorted(w / h for w, h in sizes)
    print(f"\nsize  w {min(ws)}-{max(ws)}  h {min(hs)}-{max(hs)}  "
          f"aspect {ars[0]:.2f} / {ars[len(ars)//2]:.2f} / {ars[-1]:.2f}")

    if not meta:
        print("\n6 round-trip   skipped (no boxes)")
        return

    dev = pick_device("auto")
    model, _, itos = load_model(a.ckpt, dev)
    idx = [i for i in meta][:a.roundtrip]

    full, crops, golds = [], [], []
    for i in idx:
        im = Image.open(io.BytesIO(items[i][0])).convert("L")
        full.append(im)
        crops.append(im.crop(tuple(meta[i]["box"])))
        golds.append(items[i][1])

    pf = read(model, itos, full, dev)
    pc = read(model, itos, crops, dev)
    cer_f = sum(edit(p, g) for p, g in zip(pf, golds)) / sum(len(g) for g in golds)
    cer_c = sum(edit(p, g) for p, g in zip(pc, golds)) / sum(len(g) for g in golds)
    ok_f = sum(p == g for p, g in zip(pf, golds))
    ok_c = sum(p == g for p, g in zip(pc, golds))

    print(f"\n6 round-trip   {len(idx)} samples, model={Path(a.ckpt).name}")
    print(f"   full canvas    CER {cer_f*100:6.2f}%   exact {ok_f:3d}/{len(idx)}")
    print(f"   cropped by box CER {cer_c*100:6.2f}%   exact {ok_c:3d}/{len(idx)}")
    verdict = "BOXES CORRECT" if cer_c < 0.15 else \
              "SUSPECT - crop should read almost perfectly"
    print(f"   -> {verdict}")


if __name__ == "__main__":
    main()
