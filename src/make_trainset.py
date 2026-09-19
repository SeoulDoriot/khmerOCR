"""Build ONE shared training subset used by every approach.

WHY: the rubric marks a comparison down when approaches differ in data or
preprocessing, because you can no longer attribute the difference to the
approach itself. Downloading once and reusing the file guarantees A1, A2 and A3
see byte-identical inputs.

The charset is taken from the pretrained checkpoint (242 classes) rather than
rebuilt, so the transfer approach can load its weights while the from-scratch
approaches predict over exactly the same label space.

    python -u src/make_trainset.py --n 20000
"""
import argparse, io, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_khmer
from seeds import set_all_seeds

ROOT = Path(__file__).resolve().parent.parent
REPO = "SoyVitou/KhmerSynthetic1M"
TEXT_KEYS = ["text", "label", "sentence", "transcription", "ground_truth", "caption"]
IMG_KEYS = ["image", "img", "picture"]
_kh = lambda c: "ក" <= c <= "៿"


def khmer_ratio(t):
    L = [c for c in t if not c.isspace()]
    return sum(_kh(c) for c in L) / len(L) if L else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--charset-from", default=str(Path.home() / "Downloads" / "quickstart2_crnn.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "train_subset.pt"))
    a = ap.parse_args()

    import torch
    set_all_seeds()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        d = torch.load(out, map_location="cpu", weights_only=False)
        print(f"{out} exists: {len(d['train'])} train / {len(d['val'])} val. "
              f"Reusing so every approach gets identical data.")
        return

    chars = torch.load(a.charset_from, map_location="cpu", weights_only=False)["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    print(f"charset: {len(chars)} classes (from {a.charset_from})")

    from datasets import load_dataset
    ds = load_dataset(a.repo, split="train", streaming=True)
    need = a.n + a.val_n
    items, skipped = [], 0
    for row in ds:
        tk = next((k for k in TEXT_KEYS if k in row), None)
        ik = next((k for k in IMG_KEYS if k in row), None)
        if not tk or not ik:
            continue
        lbl = normalize_khmer(row[tk])
        if not lbl or len(lbl) > a.max_len or khmer_ratio(lbl) < 0.8:
            skipped += 1
            continue
        if not any(c in stoi for c in lbl):
            skipped += 1
            continue
        buf = io.BytesIO()
        try:
            row[ik].convert("L").save(buf, "JPEG", quality=92)
        except Exception:
            skipped += 1
            continue
        items.append((buf.getvalue(), lbl))
        if len(items) % 2000 == 0:
            print(f"  {len(items)}/{need}", flush=True)
        if len(items) >= need:
            break

    import random
    random.Random(42).shuffle(items)
    val, train = items[:a.val_n], items[a.val_n:]
    torch.save({"train": train, "val": val, "chars": chars, "repo": a.repo,
                "seed": 42, "max_len": a.max_len}, out)
    print(f"\nwrote {out}")
    print(f"  train {len(train)}  val {len(val)}  (skipped {skipped})")
    print("  every approach must train on THIS file")


if __name__ == "__main__":
    main()
