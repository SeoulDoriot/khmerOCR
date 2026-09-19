"""Train ONE approach on the shared subset. Used for all approaches so the
comparison is fair.

The project rubric requires that approaches differ in architecture or training
strategy -- not in data, preprocessing or evaluation. This script fixes
everything except those two dimensions:

    A1  --arch crnn              CRNN + CTC, from scratch
    A2  --arch crnn --init CKPT  CRNN + CTC, transfer from a pretrained model
    A3  --arch attn              Attention encoder-decoder, from scratch

A1 vs A3 isolates ARCHITECTURE. A1 vs A2 isolates TRAINING STRATEGY.

Everything else is held constant: the same results/train_subset.pt, the same
242-class charset, the same preprocessing, the same seed, the same optimizer
family and schedule, the same evaluation.

Per-approach metrics required by the rubric (trainable parameters, wall-clock
training time, hardware) are printed at the end and written to the CSV, and the
train/val curve is logged every eval so learning curves can be plotted.

    python -u src/train_approach.py --arch attn --tag A3_attn_scratch
"""
import argparse, csv, io, json, platform, signal, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from seeds import set_all_seeds
from train_mac import CRNN, prep, edit, pick_device, H, W
from models import AttnOCR

ROOT = Path(__file__).resolve().parent.parent


class SubsetDS(Dataset):
    def __init__(s, items, stoi, aug):
        s.items, s.stoi, s.aug = items, stoi, aug

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        jpeg, lbl = s.items[i]
        img = prep(Image.open(io.BytesIO(jpeg)))
        a = np.array(img, dtype=np.float32)
        if s.aug:
            import random
            if random.random() < 0.4:
                a += np.random.normal(0, random.uniform(3, 12), a.shape)
            if random.random() < 0.4:
                a = (a - 128) * random.uniform(0.7, 1.3) + 128
        a = (np.clip(a, 0, 255) / 255.0 - 0.5) / 0.5
        ids = [s.stoi[c] for c in lbl if c in s.stoi] or [1]
        return torch.from_numpy(a)[None], torch.tensor(ids), lbl


def collate(b):
    im, tg, tx = zip(*b)
    return torch.stack(im), torch.cat(tg), torch.tensor([len(t) for t in tg]), list(tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["crnn", "attn"], required=True)
    ap.add_argument("--tag", required=True, help="short name, e.g. A1_crnn_scratch")
    ap.add_argument("--init", default="", help="checkpoint to start from (transfer)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: 3e-4 scratch, 5e-5 transfer")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--subset", default=str(ROOT / "results" / "train_subset.pt"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    set_all_seeds(42)
    dev = pick_device(a.device)
    lr = a.lr if a.lr is not None else (5e-5 if a.init else 3e-4)

    d = torch.load(a.subset, map_location="cpu", weights_only=False)
    chars = d["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    print(f"approach : {a.tag}   arch={a.arch}   {'transfer' if a.init else 'from scratch'}")
    print(f"data     : {len(d['train'])} train / {len(d['val'])} val, charset {len(chars)}")
    print(f"device   : {dev}   lr={lr}   batch={a.batch_size}   steps={a.steps}")

    tl = DataLoader(SubsetDS(d["train"], stoi, True), a.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)
    vl = DataLoader(SubsetDS(d["val"], stoi, False), a.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=0)

    if a.arch == "crnn":
        model = CRNN(len(chars) + 1).to(dev)
    else:
        model = AttnOCR(len(chars)).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params   : {n_params/1e6:.2f}M trainable")

    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(ck["model"])
            print(f"init     : loaded weights from {a.init} (prior CER {ck.get('cer')})")
        except Exception as e:
            raise SystemExit(f"--init incompatible with --arch {a.arch}: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=a.steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    ce = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    out_dir = ROOT / "checkpoints"; out_dir.mkdir(exist_ok=True)
    res_dir = ROOT / "results"; res_dir.mkdir(exist_ok=True)
    ckpt = out_dir / f"{a.tag}.pt"
    curve = res_dir / f"curve_{a.tag}.csv"
    with curve.open("w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss", "val_cer", "val_wer", "lr"])

    def make_targets(tg, ln):
        """attn: [BOS] x ... / x ... [EOS], char ids shifted past the 3 specials."""
        seqs = torch.split(tg, ln.tolist())
        T = int(ln.max()) + 1
        ti = torch.zeros(len(seqs), T, dtype=torch.long)
        to = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s + 2
            ti[i, 0] = AttnOCR.BOS
            ti[i, 1:len(s) + 1] = s
            to[i, :len(s)] = s
            to[i, len(s)] = AttnOCR.EOS
        return ti, to

    def decode_ctc(seq):
        o, p = [], None
        for i in seq:
            if i != 0 and i != p:
                o.append(itos.get(i, ""))
            p = i
        return "".join(o)

    @torch.no_grad()
    def ev():
        model.eval(); P, G, losses = [], [], []
        for im, tg, ln, tx in vl:
            im = im.to(dev)
            if a.arch == "crnn":
                lg = model(im)
                lp = lg.log_softmax(-1).permute(1, 0, 2).float()
                losses.append(ctc(lp, tg, torch.full((im.size(0),), lg.size(1),
                                                     dtype=torch.long), ln).item())
                P += [decode_ctc(s) for s in lg.argmax(-1).cpu().tolist()]
            else:
                ti, to = make_targets(tg, ln)
                lg = model(im, ti.to(dev))
                losses.append(ce(lg.reshape(-1, lg.size(-1)), to.to(dev).reshape(-1)).item())
                for s in model.generate(im).cpu().tolist():
                    cs = []
                    for t in s:
                        if t in (AttnOCR.EOS, AttnOCR.PAD):
                            break
                        cs.append(itos.get(t - 2, ""))
                    P.append("".join(cs))
            G += tx
        model.train()
        cer = sum(edit(p, g) for p, g in zip(P, G)) / max(sum(len(g) for g in G), 1)
        wer = sum(p != g for p, g in zip(P, G)) / len(G)
        return cer, wer, float(np.mean(losses)), list(zip(P, G))[:2]

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    print("-" * 62, flush=True)
    t_start = time.time()
    best, step, run, t0 = 9.9, 0, 0.0, time.time()
    while step < a.steps and not stop["now"]:
        for im, tg, ln, tx in tl:
            if step >= a.steps or stop["now"]:
                break
            im = im.to(dev)
            if a.arch == "crnn":
                lg = model(im)
                lp = lg.log_softmax(-1).permute(1, 0, 2).float()
                loss = ctc(lp, tg, torch.full((im.size(0),), lg.size(1), dtype=torch.long), ln)
            else:
                ti, to = make_targets(tg, ln)
                lg = model(im, ti.to(dev))
                loss = ce(lg.reshape(-1, lg.size(-1)), to.to(dev).reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sch.last_epoch < a.steps - 1:
                sch.step()
            run += loss.item(); step += 1

            if step % 50 == 0:
                print(f"step {step:5d}/{a.steps}  loss {run/50:7.4f}  "
                      f"{50*a.batch_size/(time.time()-t0):5.1f} img/s", flush=True)
                run, t0 = 0.0, time.time()
            if step % a.eval_every == 0 or step == a.steps:
                cer, wer, vloss, ex = ev()
                flag = ""
                if cer < best:
                    best = cer; flag = "  <- best"
                    torch.save({"model": model.state_dict(), "chars": chars, "cer": cer,
                                "step": step, "arch": a.arch, "tag": a.tag,
                                "params": n_params}, ckpt)
                with curve.open("a", newline="") as f:
                    csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{vloss:.4f}",
                                            f"{cer:.4f}", f"{wer:.4f}",
                                            f"{sch.get_last_lr()[0]:.2e}"])
                print(f"  >> val CER {cer:.4f}  WER {wer:.4f}  loss {vloss:.4f}{flag}")
                for p, g in ex[:1]:
                    print(f"     pred: {p}\n     gold: {g}")
                t0 = time.time()

    mins = (time.time() - t_start) / 60
    meta = {"tag": a.tag, "arch": a.arch, "strategy": "transfer" if a.init else "scratch",
            "params": n_params, "steps": step, "batch_size": a.batch_size, "lr": lr,
            "best_val_cer": best, "train_minutes": round(mins, 1),
            "device": str(dev), "hardware": platform.platform(),
            "torch": torch.__version__, "seed": 42,
            "train_n": len(d["train"]), "val_n": len(d["val"])}
    (res_dir / f"meta_{a.tag}.json").write_text(json.dumps(meta, indent=2))
    print("\n" + "=" * 62)
    print(f"{a.tag}:  best val CER {best:.4f}   {n_params/1e6:.2f}M params   "
          f"{mins:.1f} min on {dev}")
    print(f"  checkpoint {ckpt}")
    print(f"  curve      {curve}")
    print(f"  metadata   {res_dir}/meta_{a.tag}.json")


if __name__ == "__main__":
    main()
