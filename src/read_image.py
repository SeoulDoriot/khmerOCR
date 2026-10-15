"""Read Khmer text out of an image file. Point it at a cropped line.

    python -u src/read_image.py photo.jpg
    python -u src/read_image.py *.png --ckpt checkpoints/A2_crnn_transfer.pt

The model expects ONE line of text, cropped reasonably tight. A full page
will not work -- it has no detector; see src/page_ocr.py for that.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_mac import CRNN, prep, pick_device
from normalize import normalize_khmer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "checkpoints" / "v7_63800.pt"


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    m = CRNN(len(chars) + 1).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, {i + 1: c for i, c in enumerate(chars)}, ck


@torch.no_grad()
def read(model, itos, img, dev):
    x = torch.from_numpy((np.array(prep(img), dtype=np.float32) / 255. - .5) / .5)[None][None].to(dev)
    ids = model(x).argmax(-1)[0].tolist()
    out, prev = [], None
    for t in ids:
        if t != 0 and t != prev:
            out.append(itos.get(t, ""))
        prev = t
    return normalize_khmer("".join(out)), sum(1 for t in ids if t != 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default=str(DEFAULT))
    a = ap.parse_args()

    dev = pick_device("auto")
    model, itos, ck = load(a.ckpt, dev)
    print(f"model {Path(a.ckpt).name}  step {ck.get('step','-')}  "
          f"CER {ck.get('best') or ck.get('cer')}  device {dev}\n")

    for p in a.images:
        path = Path(p)
        if not path.exists():
            print(f"{path.name}: not found"); continue
        img = Image.open(path).convert("L")
        text, emitted = read(model, itos, img, dev)
        ar = img.size[0] / max(img.size[1], 1)
        warn = ""
        if ar < 3:
            warn = "   <- narrow crop; the model wants a wide single line"
        elif img.size[1] > 200:
            warn = "   <- tall image; crop closer to the text"
        print(f"{path.name}  {img.size[0]}x{img.size[1]}  aspect {ar:.1f}  "
              f"{emitted}/128 timesteps used{warn}")
        print(f"   {text or '(nothing read)'}\n")


if __name__ == "__main__":
    main()
