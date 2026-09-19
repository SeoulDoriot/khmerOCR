"""Resumable training. Colab WILL disconnect -- this is built around that.

Every checkpoint stores model + optimizer + scheduler + epoch + step, so a
killed session resumes exactly where it stopped. A run that cannot resume is
a run that never finishes on free Colab.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Charset, LMDBDataset, CurriculumSampler, collate
from models import build, AttnOCR


def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(preds, golds):
    """CER = character error rate, WER = exact-sequence error rate.

    On Khmer, WER is computed over whole crops rather than space-delimited
    words, because Khmer does not separate words with spaces.
    """
    ce = sum(edit_distance(p, g) for p, g in zip(preds, golds))
    cn = sum(len(g) for g in golds) or 1
    wrong = sum(p != g for p, g in zip(preds, golds))
    return ce / cn, wrong / max(len(golds), 1)


@torch.no_grad()
def evaluate(model, loader, charset, device, arch, limit_batches=None):
    model.eval()
    preds, golds = [], []
    for bi, (imgs, targets, lengths, widths, texts) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if arch == "crnn":
                logits = model(imgs)
                ids = logits.argmax(-1).cpu().tolist()
                preds += [charset.decode_ctc(s) for s in ids]
            else:
                seq = model.generate(imgs).cpu().tolist()
                for s in seq:
                    chars = []
                    for t in s:
                        if t in (AttnOCR.EOS, AttnOCR.PAD):
                            break
                        chars.append(charset.itos.get(t - 2, ""))
                    preds.append("".join(chars))
        golds += texts
    model.train()
    return metrics(preds, golds), list(zip(preds[:8], golds[:8]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-lmdb", nargs="+", required=True)
    ap.add_argument("--val-lmdb", nargs="+", required=True)
    ap.add_argument("--charset", required=True)
    ap.add_argument("--arch", choices=["crnn", "attn"], default="crnn")
    ap.add_argument("--out", required=True, help="checkpoint dir (put this on Drive)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--samples-per-epoch", type=int, default=None)
    ap.add_argument("--wandb", default=None, help="W&B project name; omit to disable")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} arch={args.arch}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} "
              f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    charset = Charset.load(args.charset)
    print(f"charset={len(charset)} classes")

    train_ds = LMDBDataset(args.train_lmdb, charset, augment=True)
    val_ds = LMDBDataset(args.val_lmdb, charset, augment=False)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    sampler = CurriculumSampler(train_ds.tiers,
                                samples_per_epoch=args.samples_per_epoch)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=True)

    model = build(args.arch, len(charset)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(sampler) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    ce = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # ---- resume ----
    start_epoch, gstep, best = 0, 0, 1e9
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, gstep, best = ck["epoch"], ck["step"], ck["best"]
        print(f"RESUMED from epoch {start_epoch} step {gstep} (best CER {best:.4f})")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb, config=vars(args), resume="allow")

    log_path = out / "progress.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["time", "epoch", "step", "loss", "lr", "val_cer", "val_wer"])

    def save(tag, cer):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "step": gstep, "best": best,
                    "arch": args.arch, "charset": charset.chars}, out / tag)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        print(f"\nepoch {epoch}  curriculum mix = {sampler.weights}")
        t0, running = time.time(), 0.0

        for imgs, targets, lengths, widths, texts in train_ld:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                if args.arch == "crnn":
                    logits = model(imgs)
                    logp = logits.log_softmax(-1).permute(1, 0, 2)
                    in_len = torch.full((imgs.size(0),), logits.size(1),
                                        dtype=torch.long)
                    loss = ctc(logp.float(), targets, in_len, lengths)
                else:
                    seqs = torch.split(targets, lengths.tolist())
                    T = int(lengths.max()) + 1
                    ti = torch.zeros(len(seqs), T, dtype=torch.long)
                    to = torch.zeros(len(seqs), T, dtype=torch.long)
                    for i, s in enumerate(seqs):
                        s = s + 2   # charset idx -> token id (0/1/2 reserved)
                        ti[i, 0] = AttnOCR.BOS
                        ti[i, 1:len(s) + 1] = s
                        to[i, :len(s)] = s
                        to[i, len(s)] = AttnOCR.EOS
                    ti, to = ti.to(device), to.to(device)
                    logits = model(imgs, ti)
                    loss = ce(logits.reshape(-1, logits.size(-1)), to.reshape(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()

            running += loss.item()
            gstep += 1

            if gstep % 50 == 0:
                avg = running / 50
                running = 0.0
                ips = 50 * args.batch_size / (time.time() - t0)
                t0 = time.time()
                print(f"  step {gstep} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{ips:.0f} img/s", flush=True)
                if run:
                    run.log({"loss": avg, "lr": sched.get_last_lr()[0], "step": gstep})

            if gstep % args.save_every == 0:
                save("last.pt", best)

        (cer, wer), samples = evaluate(model, val_ld, charset, device, args.arch,
                                       limit_batches=20)
        print(f"epoch {epoch}: val CER {cer:.4f}  WER {wer:.4f}")
        for p, g in samples[:4]:
            print(f"    pred={p!r}\n    gold={g!r}")
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%F %T"), epoch, gstep,
                                    f"{avg:.4f}", f"{sched.get_last_lr()[0]:.2e}",
                                    f"{cer:.4f}", f"{wer:.4f}"])
        if run:
            run.log({"val_cer": cer, "val_wer": wer, "epoch": epoch})
        save("last.pt", best)
        if cer < best:
            best = cer
            save("best.pt", best)
            print(f"    new best CER {best:.4f}")

    print(f"\ndone. best val CER {best:.4f}  checkpoints in {out}")


if __name__ == "__main__":
    main()
