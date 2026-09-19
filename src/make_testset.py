"""Build ONE frozen test set that every approach is evaluated on.

WHY THIS EXISTS: the project rubric requires "the same held-out test data so
that the comparison is fair", and marks a comparison down when preprocessing or
splits differ between approaches. Until now each experiment used its own split,
so the numbers were not comparable.

This samples a fixed, seeded slice from each corpus, stores the raw JPEG bytes
plus the normalized label, and writes a single file. Once built it is NEVER
regenerated -- every approach loads this exact file.

The handwriting portion reproduces the split used by finetune_handwriting.py
(seed 42, first 20% of the shuffled index), so those samples were never trained
on by any approach.

    python -u src/make_testset.py --out results/test_set.pt
"""
import argparse
import io
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_khmer
from seeds import set_all_seeds

ROOT = Path(__file__).resolve().parent.parent
TEXT_KEYS = ["text", "label", "sentence", "transcription", "ground_truth", "caption"]
IMG_KEYS = ["image", "img", "picture"]
_kh = lambda c: "ក" <= c <= "៿"


def khmer_ratio(t):
    L = [c for c in t if not c.isspace()]
    return sum(_kh(c) for c in L) / len(L) if L else 0.0


# (repo, domain label, how many, whether it was ever trained on by any approach)
SOURCES = [
    ("Chanrith123333/khmer_english_ocr_image_line", "printed_A", 300, True),
    ("SoyVitou/KhmerSynthetic1M",                   "printed_B", 300, True),
    ("SoyVitou/62k-images-khmer-printed-dataset",   "printed_C", 200, False),
    ("SoyVitou/khmer-handwritten-dataset-4.2k",     "handwriting", 0, False),  # held-out split
    ("SoyVitou/KhmerST-Dataset-OCR-Cropped",        "camera",      200, False),
]


def collect(repo, n, max_len, khmer_only, skip=0):
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    out, seen = [], 0
    for row in ds:
        tk = next((k for k in TEXT_KEYS if k in row), None)
        ik = next((k for k in IMG_KEYS if k in row), None)
        if not tk or not ik:
            continue
        lbl = normalize_khmer(row[tk])
        if not lbl or len(lbl) > max_len:
            continue
        if khmer_only and khmer_ratio(lbl) < 0.8:
            continue
        seen += 1
        if seen <= skip:
            continue
        buf = io.BytesIO()
        try:
            row[ik].convert("L").save(buf, "JPEG", quality=95)
        except Exception:
            continue
        out.append((buf.getvalue(), lbl))
        if len(out) >= n:
            break
    return out


def handwriting_heldout(max_len, khmer_only):
    """Reproduce finetune_handwriting.py's split exactly: seed 42, first 20%."""
    from datasets import load_dataset
    ds = load_dataset("SoyVitou/khmer-handwritten-dataset-4.2k", split="train", streaming=True)
    rows = []
    for row in ds:
        tk = next((k for k in TEXT_KEYS if k in row), None)
        ik = next((k for k in IMG_KEYS if k in row), None)
        if not tk or not ik:
            continue
        lbl = normalize_khmer(row[tk])
        if not lbl or len(lbl) > max_len:
            continue
        rows.append((row[ik], lbl))
    rng = random.Random(42)                    # MUST match the fine-tuning script
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.2)
    out = []
    for i in idx[:cut]:
        img, lbl = rows[i]
        buf = io.BytesIO()
        try:
            img.convert("L").save(buf, "JPEG", quality=95)
        except Exception:
            continue
        out.append((buf.getvalue(), lbl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "test_set.pt"))
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--khmer-only", action="store_true", default=True)
    a = ap.parse_args()

    import torch
    set_all_seeds()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} already exists. The test set is frozen -- refusing to "
              f"regenerate it. Delete it deliberately if you really mean to.")
        return

    items, summary = [], {}
    for repo, domain, n, trained_on in SOURCES:
        print(f"\n=== {domain}  ({repo})", flush=True)
        try:
            if domain == "handwriting":
                got = handwriting_heldout(a.max_len, a.khmer_only)
            else:
                got = collect(repo, n, a.max_len, a.khmer_only)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        for b, l in got:
            items.append({"jpeg": b, "text": l, "domain": domain,
                          "seen_in_training": trained_on})
        summary[domain] = {"n": len(got), "repo": repo, "seen_in_training": trained_on}
        print(f"  {len(got)} samples")

    torch.save({"items": items, "summary": summary, "seed": 42,
                "max_len": a.max_len, "khmer_only": a.khmer_only}, out)
    print("\n" + "=" * 60)
    print(f"FROZEN TEST SET -> {out}   ({len(items)} samples)")
    for d, s in summary.items():
        flag = "" if s["seen_in_training"] else "   [unseen domain]"
        print(f"  {s['n']:5d}  {d}{flag}")
    print("\nEvery approach must be evaluated on THIS file. Do not regenerate it.")


if __name__ == "__main__":
    main()
