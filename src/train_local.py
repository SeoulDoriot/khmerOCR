"""Train from a LOCALLY CACHED dataset instead of streaming. For the Mac.

WHY: streaming five corpora over a home connection spent 30 minutes at 1% CPU
doing nothing but re-downloading 180MB parquet shards and timing out. The GPU
sat idle. Streaming suits Colab (fast datacentre network); it does not suit a
laptop on home wifi.

This downloads ONE dataset once (~2.6GB, resumable, cached by HuggingFace),
then trains from local disk at full speed with no further network use. Arrow
files are memory-mapped, so a 100k-image dataset costs almost no RAM.

Same model and checkpoint format as train_mac.py.

    python -u src/train_local.py --steps 40000
"""
import argparse, os, random, signal, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_khmer
from train_mac import CRNN, prep, to_tensor, edit, pick_device, H, W, TEXT_KEYS, IMG_KEYS

ROOT = Path(__file__).resolve().parent.parent
_khmer = lambda c: "ក" <= c <= "៿"

def khmer_ratio(t):
    L = [c for c in t if not c.isspace()]
    return sum(_khmer(c) for c in L) / len(L) if L else 0.0


class ArrowDS(Dataset):
    """Indexes a memory-mapped HF dataset. Rows are decoded lazily per item."""
    def __init__(s, ds, idx, tk, ik, stoi, aug):
        s.ds, s.idx, s.tk, s.ik, s.stoi, s.aug = ds, idx, tk, ik, stoi, aug
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        row = s.ds[int(s.idx[i])]
        lbl = normalize_khmer(row[s.tk])
        ids = [s.stoi[c] for c in lbl if c in s.stoi]
        if not ids:
            ids = [1]; lbl = ""
        return to_tensor(prep(row[s.ik]), aug=s.aug), torch.tensor(ids), lbl


def collate(b):
    im, tg, tx = zip(*b)
    return torch.stack(im), torch.cat(tg), torch.tensor([len(t) for t in tg]), list(tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Sokheng/khmer-synthetic-ocr-v1-100k")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--khmer-only", action="store_true")
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "local_state.pt"))
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()

    dev = pick_device(a.device)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    best_path = str(out) + ".best.pt"
    print(f"device={dev}  batch={a.batch_size}")

    from datasets import load_dataset
    import glob as _glob
    t0 = time.time()
    if ".parquet" in a.repo or a.repo.startswith("/"):
        # local parquet shards: lets us train on what has finished downloading
        files = sorted(_glob.glob(a.repo))
        if not files:
            raise SystemExit(f"no files matched {a.repo}")
        print(f"loading {len(files)} local parquet file(s) ...", flush=True)
        for f in files:
            print(f"   {os.path.getsize(f)/1e6:.0f} MB  {os.path.basename(f)}")
        ds = load_dataset("parquet", data_files=files, split="train")
    else:
        print(f"downloading/loading {a.repo} (cached after first time) ...", flush=True)
        ds = load_dataset(a.repo, split="train")
    print(f"  {len(ds)} rows in {time.time()-t0:.0f}s  (memory-mapped, low RAM)")

    first = ds[0]
    tk = next((k for k in TEXT_KEYS if k in first), None)
    ik = next((k for k in IMG_KEYS if k in first), None)
    print(f"  columns: text={tk} image={ik}")
    if not tk or not ik:
        raise SystemExit(f"columns not found: {list(first)}")

    print("scanning labels ...", flush=True)
    texts = ds[tk]
    keep, chars = [], {}
    from collections import Counter
    cnt = Counter()
    for i, raw in enumerate(texts):
        lbl = normalize_khmer(raw)
        if not lbl or len(lbl) > a.max_len: continue
        if a.khmer_only and khmer_ratio(lbl) < 0.8: continue
        keep.append(i); cnt.update(lbl)
    chars = sorted(c for c, k in cnt.items() if k >= 5)
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    print(f"  kept {len(keep)}/{len(ds)}   charset={len(chars)}")

    rng = random.Random(0); rng.shuffle(keep)
    val_idx, train_idx = keep[:a.val_size], keep[a.val_size:]
    print(f"  train={len(train_idx)} val={len(val_idx)}")

    tl = DataLoader(ArrowDS(ds, train_idx, tk, ik, stoi, True), a.batch_size,
                    shuffle=True, collate_fn=collate, num_workers=a.workers,
                    persistent_workers=a.workers > 0, drop_last=True)
    vl = DataLoader(ArrowDS(ds, val_idx, tk, ik, stoi, False), a.batch_size,
                    shuffle=False, collate_fn=collate, num_workers=0)

    start_step, best, rc = 0, 9.9, 0
    resuming = out.exists()
    ck = None
    if resuming:
        ck = torch.load(out, map_location="cpu", weights_only=False)
        if len(ck["chars"]) == len(chars):
            start_step = ck.get("step", 0); best = ck.get("best", 9.9)
            rc = ck.get("resume_count", 0) + 1
            print(f"RESUMING step {start_step}/{a.steps} best {best:.4f}")
        else:
            print(f"checkpoint charset {len(ck['chars'])} != {len(chars)}; starting fresh")
            ck = None
    if start_step >= a.steps:
        print("target reached."); return

    model = CRNN(len(chars) + 1).to(dev)
    if ck is not None:
        try: model.load_state_dict(ck["model"]); print("  weights loaded")
        except Exception as e: print(f"  fresh weights ({e})")
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    if ck is not None and "opt" in ck:
        try:
            opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sched"])
            print(f"  optimizer restored (lr {sch.get_last_lr()[0]:.2e})")
        except Exception: pass
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def decode(seq):
        o, prev = [], None
        for i in seq:
            if i != 0 and i != prev: o.append(itos.get(i, ""))
            prev = i
        return "".join(o)

    @torch.no_grad()
    def ev():
        model.eval(); P, G = [], []
        for im, tg, ln, tx in vl:
            lg = model(im.to(dev))
            P += [decode(s) for s in lg.argmax(-1).cpu().tolist()]; G += tx
        model.train()
        return (sum(edit(p, g) for p, g in zip(P, G)) / max(sum(len(g) for g in G), 1),
                sum(p != g for p, g in zip(P, G)) / len(G), list(zip(P, G))[:2])

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sch.state_dict(), "chars": chars, "step": step,
                    "best": best, "resume_count": rc}, out)

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    print(f"\ntraining to {a.steps} (from {start_step})\n" + "-" * 60, flush=True)
    step, run, t0 = start_step, 0.0, time.time()
    while step < a.steps and not stop["now"]:
        for im, tg, ln, tx in tl:
            if step >= a.steps or stop["now"]: break
            im = im.to(dev)
            lg = model(im)
            lp = lg.log_softmax(-1).permute(1, 0, 2).float()
            loss = ctc(lp, tg, torch.full((im.size(0),), lg.size(1), dtype=torch.long), ln)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sch.last_epoch < a.steps - 1: sch.step()
            run += loss.item(); step += 1
            if step % 50 == 0:
                ips = 50 * a.batch_size / (time.time() - t0)
                eta = (a.steps - step) * a.batch_size / max(ips, 1) / 3600
                print(f"step {step:6d}/{a.steps}  loss {run/50:7.4f}  "
                      f"{ips:5.1f} img/s  eta {eta:.1f}h", flush=True)
                run, t0 = 0.0, time.time()
            if step % a.save_every == 0: save(step)
            if step % a.eval_every == 0 or step == a.steps:
                cer, wer, ex = ev()
                flag = ""
                if cer < best:
                    best = cer; flag = "  <- best"
                    torch.save({"model": model.state_dict(), "chars": chars,
                                "cer": cer, "step": step}, best_path)
                save(step)
                print(f"  >> CER {cer:.4f}  WER {wer:.4f}{flag}", flush=True)
                for p, g in ex[:1]:
                    print(f"     pred: {p}\n     gold: {g}")
                t0 = time.time()

    save(step)
    print(f"\nstopped at step {step}. best CER {best:.4f}\n{out}\n{best_path}")


if __name__ == "__main__":
    main()
