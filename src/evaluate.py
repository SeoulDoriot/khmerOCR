"""Evaluate every trained model on the FROZEN test set. One table, one protocol.

The rubric requires all approaches to be scored on identical held-out data with
identical test-time preprocessing. This script is the only place evaluation
happens, so that requirement is enforced by construction rather than by care.

Reports CER and WER overall and per domain. Three of the five domains were
never trained on by any approach, which separates generalisation from
memorisation.

    python -u src/evaluate.py
"""
import argparse, io, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from seeds import set_all_seeds
from train_mac import CRNN, prep, edit, pick_device
from models import AttnOCR
from normalize import normalize_khmer

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("A1_crnn_scratch",   "checkpoints/A1_crnn_scratch.pt",   "crnn", "CRNN+CTC, from scratch"),
    ("A2_crnn_transfer",  "checkpoints/A2_crnn_transfer.pt",  "crnn", "CRNN+CTC, transfer"),
    ("A3_attn_scratch",   "checkpoints/A3_attn_scratch.pt",   "attn", "Attention enc-dec, scratch"),
    ("handwriting",       "checkpoints/handwriting.pt",       "crnn", "A2 + handwriting fine-tune"),
    ("baseline_pretrained", str(Path.home()/"Downloads"/"quickstart2_crnn.pt"), "crnn",
     "Pretrained printed specialist"),
    ("v7_63800",           "checkpoints/v7_63800.pt",          "crnn", "Single-corpus, 63.8k steps"),
    ("v8_28800",           "checkpoints/v8_28800.pt",          "crnn", "v7 + 28.8k joint multi-corpus steps"),
]


def load(path, arch, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    model = (CRNN(len(chars) + 1) if arch == "crnn" else AttnOCR(len(chars))).to(dev).eval()
    model.load_state_dict(ck["model"])
    return model, chars, ck


@torch.no_grad()
def run(model, chars, arch, items, dev, bs=48):
    itos = {i + 1: c for i, c in enumerate(chars)}
    preds, golds = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = torch.stack([
            torch.from_numpy((np.array(prep(Image.open(io.BytesIO(it["jpeg"]))),
                                       dtype=np.float32) / 255.0 - 0.5) / 0.5)[None]
            for it in chunk]).to(dev)
        if arch == "crnn":
            lg = model(x)
            for s in lg.argmax(-1).cpu().tolist():
                o, p = [], None
                for t in s:
                    if t != 0 and t != p:
                        o.append(itos.get(t, ""))
                    p = t
                preds.append("".join(o))
        else:
            for s in model.generate(x).cpu().tolist():
                o = []
                for t in s:
                    if t in (AttnOCR.EOS, AttnOCR.PAD):
                        break
                    o.append(itos.get(t - 2, ""))
                preds.append("".join(o))
        golds += [it["text"] for it in chunk]
    cer = sum(edit(p, g) for p, g in zip(preds, golds)) / max(sum(len(g) for g in golds), 1)
    wer = sum(p != g for p, g in zip(preds, golds)) / max(len(golds), 1)
    # Khmer encodes visually identical text several ways; normalizing both sides
    # measures recognition instead of codepoint ordering. Reported separately so
    # the raw number stays visible.
    npreds = [normalize_khmer(x) for x in preds]
    ngolds = [normalize_khmer(x) for x in golds]
    ncer = sum(edit(p, g) for p, g in zip(npreds, ngolds)) / max(sum(len(g) for g in ngolds), 1)
    nwer = sum(p != g for p, g in zip(npreds, ngolds)) / max(len(ngolds), 1)
    return cer, wer, preds, golds, {"cer_norm": ncer, "wer_norm": nwer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=str(ROOT / "results" / "test_set.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "evaluation.json"))
    ap.add_argument("--only", default=None,
                    help="comma-separated model tags to score (default: all)")
    a = ap.parse_args()

    set_all_seeds(42)
    dev = pick_device("auto")
    ts = torch.load(a.test_set, map_location="cpu", weights_only=False)
    items = ts["items"]
    domains = sorted({it["domain"] for it in items})
    unseen = {d for d in domains
              if not any(it["seen_in_training"] for it in items if it["domain"] == d)}
    print(f"frozen test set: {len(items)} samples, {len(domains)} domains")
    print(f"never trained on: {', '.join(sorted(unseen))}\n")

    keep = set(a.only.split(",")) if a.only else None
    table = {}
    for tag, path, arch, desc in MODELS:
        if keep and tag not in keep:
            continue
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"[skip] {tag}: no checkpoint at {p}")
            continue
        model, chars, ck = load(p, arch, dev)
        meta_p = ROOT / "results" / f"meta_{tag}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        row = {"desc": desc, "arch": arch,
               "params": meta.get("params", sum(x.numel() for x in model.parameters())),
               "train_minutes": meta.get("train_minutes"),
               "device": meta.get("device", str(dev)), "domains": {}}
        cer_all, wer_all, _, _, nrm = run(model, chars, arch, items, dev)
        row["overall"] = {"cer": cer_all, "wer": wer_all, **nrm}
        for d in domains:
            sub = [it for it in items if it["domain"] == d]
            c, w, _, _, dn = run(model, chars, arch, sub, dev)
            row["domains"][d] = {"cer": c, "wer": w, "n": len(sub),
                                 "seen_in_training": sub[0]["seen_in_training"], **dn}
        table[tag] = row
        print(f"{tag:<22} CER {cer_all:.4f}  WER {wer_all:.4f}"
              f"  (normalized CER {nrm['cer_norm']:.4f})")

    Path(a.out).write_text(json.dumps({"test_set": a.test_set, "n": len(items),
                                       "results": table}, indent=2))

    print("\n" + "=" * 96)
    hdr = f"{'model':<22}{'params':>9}{'min':>6}{'CER':>8}{'WER':>8}"
    for d in domains:
        hdr += f"{d[:11]:>13}"
    print(hdr)
    print("-" * 96)
    for tag, r in table.items():
        line = (f"{tag:<22}{r['params']/1e6:>8.2f}M"
                f"{(r['train_minutes'] or 0):>6.0f}{r['overall']['cer']:>8.4f}"
                f"{r['overall']['wer']:>8.4f}")
        for d in domains:
            line += f"{r['domains'][d]['cer']:>13.4f}"
        print(line)
    print("-" * 96)
    print("domains never trained on: " + ", ".join(sorted(unseen)))
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
