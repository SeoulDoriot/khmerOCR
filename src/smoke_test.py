"""End-to-end smoke test on synthetic data. No Colab, no GPU, no downloads.

Proves: LMDB write/read, charset encode/decode, curriculum sampler, collate,
both architectures forward+backward, CTC and CE losses, and greedy decoding.
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_khmer
from dataset import Charset, LMDBDataset, CurriculumSampler, collate
from models import build, AttnOCR
from torch.utils.data import DataLoader

TMP = Path("/tmp/khmer_smoke")
KH = [chr(c) for c in range(0x1780, 0x17A3)]   # base consonants
VOW = [chr(c) for c in range(0x17B6, 0x17C6)]  # dependent vowels


def make_label(tier):
    n = {"char": 1, "word": random.randint(3, 8),
         "sentence": random.randint(15, 25)}[tier]
    s = ""
    for _ in range(n):
        s += random.choice(KH)
        if random.random() < 0.4:
            s += random.choice(VOW)
    return normalize_khmer(s[:n])


def build_lmdb(path, n=300):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    env = lmdb.open(str(path), map_size=1 << 30)
    tiers = ["char"] * (n // 3) + ["word"] * (n // 3) + ["sentence"] * (n - 2 * (n // 3))
    with env.begin(write=True) as txn:
        for i, tier in enumerate(tiers):
            lbl = make_label(tier) or KH[0]
            w = max(16, len(lbl) * 20)
            img = Image.new("L", (w, 64), 255)
            ImageDraw.Draw(img).rectangle([2, 10, w - 2, 54], outline=0)
            import io
            buf = io.BytesIO()
            img.save(buf, "JPEG")
            txn.put(f"img-{i:09d}".encode(), buf.getvalue())
            txn.put(f"lbl-{i:09d}".encode(), lbl.encode())
            txn.put(f"tier-{i:09d}".encode(), tier.encode())
        txn.put(b"num-samples", str(n).encode())
    env.close()
    return path


def main():
    random.seed(0); torch.manual_seed(0)
    TMP.mkdir(parents=True, exist_ok=True)

    print("1. building synthetic LMDBs")
    train_p = build_lmdb(TMP / "train", 300)
    val_p = build_lmdb(TMP / "val", 60)

    print("2. charset")
    chars = sorted(set("".join(KH + VOW)))
    cs_path = TMP / "charset.json"
    cs_path.write_text(json.dumps({"charset": chars, "size": len(chars)}))
    charset = Charset.load(cs_path)
    assert charset.decode_ctc(charset.encode("កា")) == "កា", "ctc roundtrip failed"
    print(f"   {len(charset)} classes, encode/decode roundtrip OK")

    print("3. dataset + curriculum sampler")
    ds = LMDBDataset([train_p], charset, augment=True)
    sampler = CurriculumSampler(ds.tiers, epoch=0)
    from collections import Counter
    mix = Counter(ds.tiers[i] for i in list(sampler))
    tot = sum(mix.values())
    print(f"   epoch0 mix: {dict((k, round(v / tot, 2)) for k, v in mix.items())}")
    sampler.set_epoch(9)
    mix9 = Counter(ds.tiers[i] for i in list(sampler))
    t9 = sum(mix9.values())
    print(f"   epoch9 mix: {dict((k, round(v/t9, 2)) for k, v in mix9.items())}")
    assert mix["char"] / tot > mix9["char"] / t9, "curriculum did not shift"

    print("4. forward/backward both architectures")
    for arch in ["crnn", "attn"]:
        sampler.set_epoch(0)
        ld = DataLoader(ds, batch_size=8, sampler=sampler, collate_fn=collate,
                        num_workers=0, drop_last=True)
        model = build(arch, len(charset))
        n_p = sum(p.numel() for p in model.parameters())
        imgs, targets, lengths, widths, texts = next(iter(ld))
        if arch == "crnn":
            logits = model(imgs)
            logp = logits.log_softmax(-1).permute(1, 0, 2)
            in_len = torch.full((imgs.size(0),), logits.size(1), dtype=torch.long)
            loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(logp, targets, in_len, lengths)
            assert logits.size(1) >= int(lengths.max()), \
                f"CTC needs T>=label_len; T={logits.size(1)} maxlab={int(lengths.max())}"
        else:
            seqs = torch.split(targets, lengths.tolist())
            T = int(lengths.max()) + 1
            ti = torch.zeros(len(seqs), T, dtype=torch.long)
            to = torch.zeros(len(seqs), T, dtype=torch.long)
            for i, s in enumerate(seqs):
                s = s + 2
                ti[i, 0] = AttnOCR.BOS
                ti[i, 1:len(s) + 1] = s
                to[i, :len(s)] = s
                to[i, len(s)] = AttnOCR.EOS
            logits = model(imgs, ti)
            loss = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)(
                logits.reshape(-1, logits.size(-1)), to.reshape(-1))
        loss.backward()
        gn = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert gn > 0, f"{arch}: zero gradient"
        print(f"   {arch}: {n_p/1e6:.1f}M params, loss {loss.item():.4f}, grads flowing")

    print("5. greedy decode (attn)")
    model = build("attn", len(charset)).eval()
    with torch.no_grad():
        seq = model.generate(imgs[:2], max_len=12)
    print(f"   generate -> shape {tuple(seq.shape)}")

    print("\nALL CHECKS PASSED")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
