"""Stage 2: adapt a printed-text Khmer OCR model to real handwriting.

THE PROBLEM THIS SOLVES (domain shift): stage 1 learned Khmer glyphs from
rendered fonts. Handwriting is a different visual distribution -- variable
stroke width, slant, spacing, and no two samples identical. A model that has
seen zero handwriting cannot read handwriting, however good it is on print.

WHY FINE-TUNING AND NOT RETRAINING: stage 1 already knows what Khmer characters
ARE. Stage 2 only has to teach a new visual style, which needs far less data.
Training from scratch on 3,360 handwriting samples would fail badly.

THREE THINGS THAT KEEP THIS FROM DESTROYING THE MODEL:

  1. REPLAY. 4.2k samples is tiny; training only on them causes catastrophic
     forgetting -- the model memorises the handwriting and loses printed text.
     --replay 0.3 keeps 30% synthetic printed data in every batch.
  2. LOW LEARNING RATE. 1e-5, not 5e-5. We are nudging a converged model, not
     retraining it. Higher LR knocks it out of the solution stage 1 found.
  3. EARLY STOPPING. With this little data, validation CER bottoms out then
     climbs as overfitting starts. --patience stops at the bottom.

THE SPLIT IS SACRED: 20% of the handwriting is held out and never trained on.
Without it you cannot tell adaptation from memorisation -- the exact mistake
that made the overnight Mac run report 99% on data it had memorised and 74%
on data it had not.

    python -u src/finetune_handwriting.py --init <stage1.pt> --steps 3000
"""
import argparse, math, random, signal, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_khmer
from train_mac import CRNN, prep, edit, pick_device, H, W, TEXT_KEYS, IMG_KEYS

ROOT = Path(__file__).resolve().parent.parent
HAND = "SoyVitou/khmer-handwritten-dataset-4.2k"
PRINT_REPLAY = "SoyVitou/KhmerSynthetic1M"


def aug_handwriting(img):
    """Stronger than the printed-text augmentation: real handwriting varies in
    slant, stroke weight and baseline far more than rendered fonts, and we have
    only ~3k samples to cover that variety."""
    a = np.array(img, dtype=np.float32)
    if random.random() < 0.5:                       # slant / rotation
        ang = random.uniform(-4, 4)
        img = Image.fromarray(a.astype(np.uint8)).rotate(
            ang, resample=Image.BILINEAR, fillcolor=255)
        a = np.array(img, dtype=np.float32)
    if random.random() < 0.4:                       # stroke weight
        im = Image.fromarray(a.astype(np.uint8))
        im = im.filter(ImageFilter.MinFilter(3) if random.random() < 0.5
                       else ImageFilter.MaxFilter(3))
        a = np.array(im, dtype=np.float32)
    if random.random() < 0.4:                       # blur (pen bleed / focus)
        im = Image.fromarray(a.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
        a = np.array(im, dtype=np.float32)
    if random.random() < 0.4:                       # paper noise
        a += np.random.normal(0, random.uniform(4, 14), a.shape)
    if random.random() < 0.4:                       # ink / paper contrast
        a = (a - 128) * random.uniform(0.65, 1.35) + 128 + random.uniform(-20, 20)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def to_tensor(img):
    a = np.array(img, dtype=np.float32)
    return torch.from_numpy((a / 255.0 - 0.5) / 0.5)[None]


class ListDS(Dataset):
    def __init__(s, items, stoi, aug):
        s.items, s.stoi, s.aug = items, stoi, aug
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        img, lbl = s.items[i]
        img = prep(img)
        if s.aug: img = aug_handwriting(img)
        ids = [s.stoi[c] for c in lbl if c in s.stoi] or [1]
        return to_tensor(img), torch.tensor(ids), lbl


def collate(b):
    im, tg, tx = zip(*b)
    return torch.stack(im), torch.cat(tg), torch.tensor([len(t) for t in tg]), list(tx)


def load_rows(repo, stoi, max_len, limit=None):
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    out = []
    for row in ds:
        tk = next((k for k in TEXT_KEYS if k in row), None)
        ik = next((k for k in IMG_KEYS if k in row), None)
        if not tk or not ik: continue
        lbl = normalize_khmer(row[tk])
        if not lbl or len(lbl) > max_len: continue
        try: out.append((row[ik].convert("L"), lbl))
        except Exception: continue
        if limit and len(out) >= limit: break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="stage-1 checkpoint")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--replay", type=float, default=0.3,
                    help="fraction of each batch drawn from printed data")
    ap.add_argument("--replay-size", type=int, default=4000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5,
                    help="evals without improvement before stopping")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "handwriting.pt"))
    a = ap.parse_args()

    dev = pick_device(a.device)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={dev}")

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    print(f"stage-1 checkpoint: {a.init}")
    print(f"  charset={len(chars)}  prior CER={ck.get('cer')}")

    print(f"\nloading handwriting ({HAND}) ...", flush=True)
    hand = load_rows(HAND, stoi, a.max_len)
    print(f"  {len(hand)} usable samples")

    # characters the model has no output class for -- a hard ceiling on accuracy
    oov = {c for _, l in hand for c in l if c not in stoi}
    tot = sum(len(l) for _, l in hand)
    n_oov = sum(1 for _, l in hand for c in l if c not in stoi)
    print(f"  OOV: {n_oov}/{tot} chars ({100*n_oov/max(tot,1):.2f}%) "
          f"across {len(oov)} distinct")

    rng = random.Random(42)          # FIXED seed: the split must be reproducible
    idx = list(range(len(hand))); rng.shuffle(idx)
    cut = int(len(idx) * a.test_frac)
    test = [hand[i] for i in idx[:cut]]
    train = [hand[i] for i in idx[cut:]]
    print(f"  train {len(train)}  |  held-out test {len(test)} (never trained on)")

    replay = []
    if a.replay > 0:
        print(f"\nloading replay data ({PRINT_REPLAY}, {a.replay_size}) ...", flush=True)
        replay = load_rows(PRINT_REPLAY, stoi, a.max_len, limit=a.replay_size)
        print(f"  {len(replay)} printed samples for replay")

    n_replay = int(a.batch_size * a.replay)
    n_hand = a.batch_size - n_replay
    print(f"\nbatch mix: {n_hand} handwriting + {n_replay} printed")

    hand_ld = DataLoader(ListDS(train, stoi, True), n_hand, shuffle=True,
                         collate_fn=collate, num_workers=0, drop_last=True)
    rep_ld = (DataLoader(ListDS(replay, stoi, True), n_replay, shuffle=True,
                         collate_fn=collate, num_workers=0, drop_last=True)
              if replay and n_replay else None)
    test_ld = DataLoader(ListDS(test, stoi, False), a.batch_size, shuffle=False,
                         collate_fn=collate, num_workers=0)

    model = CRNN(len(chars) + 1).to(dev)
    model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def decode(seq):
        o, p = [], None
        for i in seq:
            if i != 0 and i != p: o.append(itos.get(i, ""))
            p = i
        return "".join(o)

    @torch.no_grad()
    def ev():
        model.eval(); P, G = [], []
        for im, tg, ln, tx in test_ld:
            lg = model(im.to(dev))
            P += [decode(s) for s in lg.argmax(-1).cpu().tolist()]; G += tx
        model.train()
        return (sum(edit(p, g) for p, g in zip(P, G)) / max(sum(len(g) for g in G), 1),
                sum(p != g for p, g in zip(P, G)) / len(G), list(zip(P, G))[:3])

    cer0, wer0, ex0 = ev()
    print(f"\nBEFORE fine-tuning, on held-out handwriting:")
    print(f"  CER {cer0:.4f}  WER {wer0:.4f}")
    for p, g in ex0[:2]:
        print(f"    pred: {p}\n    gold: {g}")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    best, bad, step = cer0, 0, 0
    torch.save({"model": model.state_dict(), "chars": chars, "cer": cer0,
                "step": 0, "stage": "handwriting"}, out)
    print(f"\nfine-tuning (lr {a.lr}, replay {a.replay}, patience {a.patience})")
    print("-" * 60, flush=True)
    rep_iter = iter(rep_ld) if rep_ld else None
    run, t0 = 0.0, time.time()
    while step < a.steps and not stop["now"]:
        for im, tg, ln, tx in hand_ld:
            if step >= a.steps or stop["now"]: break
            if rep_iter is not None:
                try: rim, rtg, rln, _ = next(rep_iter)
                except StopIteration:
                    rep_iter = iter(rep_ld); rim, rtg, rln, _ = next(rep_iter)
                im = torch.cat([im, rim]); tg = torch.cat([tg, rtg])
                ln = torch.cat([ln, rln])
            im = im.to(dev)
            lg = model(im)
            lp = lg.log_softmax(-1).permute(1, 0, 2).float()
            loss = ctc(lp, tg, torch.full((im.size(0),), lg.size(1), dtype=torch.long), ln)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item(); step += 1

            if step % 50 == 0:
                print(f"step {step:5d}/{a.steps}  loss {run/50:7.4f}  "
                      f"{50*im.size(0)/(time.time()-t0):5.1f} img/s", flush=True)
                run, t0 = 0.0, time.time()
            if step % a.eval_every == 0:
                cer, wer, ex = ev()
                if cer < best:
                    best, bad = cer, 0
                    torch.save({"model": model.state_dict(), "chars": chars,
                                "cer": cer, "step": step, "stage": "handwriting"}, out)
                    print(f"  >> CER {cer:.4f}  WER {wer:.4f}  <- best (saved)")
                else:
                    bad += 1
                    print(f"  >> CER {cer:.4f}  WER {wer:.4f}  "
                          f"(no improvement {bad}/{a.patience})")
                    if bad >= a.patience:
                        print("\nearly stop: held-out CER stopped improving. "
                              "More steps would only overfit.")
                        stop["now"] = True
                        break
                t0 = time.time()

    print("\n" + "=" * 60)
    print(f"handwriting CER  {cer0:.4f}  ->  {best:.4f}")
    print(f"  improvement: {100*(cer0-best)/max(cer0,1e-9):.1f}% relative")
    print(f"  saved: {out}")
    print(f"  measured on {len(test)} held-out samples never trained on")


if __name__ == "__main__":
    main()
