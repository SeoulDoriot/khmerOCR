"""Khmer OCR training on Apple Silicon (MPS). Streams data, fully resumable.

Same model and preprocessing as the Colab scripts, so checkpoints are
interchangeable -- the state dict uses cnn/rnn/fc keys exactly as quickstart*.py.

Differences forced by the platform:
  * MPS has no working fp16 autocast or GradScaler, so this trains in fp32.
    Expect roughly 3-5x slower than a T4, but with no quota and no disconnects.
  * num_workers defaults to 0: macOS spawns (not forks) workers, and HF
    streaming iterators do not survive being pickled into a spawned process.

NOTE: training here does NOT combine with a Colab run. Two machines produce two
divergent models and there is no way to merge them. Pick one.

    python src/train_mac.py --steps 40000
    # Ctrl-C any time; run the same line again to continue.
"""
import argparse, json, os, random, signal, sys, time, unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_khmer

ROOT = Path(__file__).resolve().parent.parent
_khmer = lambda c: "ក" <= c <= "៿"

def khmer_ratio(t):
    L = [c for c in t if not c.isspace()]
    return sum(_khmer(c) for c in L) / len(L) if L else 0.0

DEFAULT_REPOS = [
    "Chanrith123333/khmer_english_ocr_image_line",
    "SoyVitou/KhmerSynthetic1M",
    "Sokheng/khmer-synthetic-ocr-v1-100k",
    "SoyVitou/62k-images-khmer-printed-dataset",
    "KimkosalYon/khmer-ocr-200k-siemreap-arial",
]
DEFAULT_WEIGHTS = [0.45, 0.25, 0.12, 0.10, 0.08]

H, W = 64, 512
TEXT_KEYS = ["text", "label", "sentence", "transcription", "ground_truth", "caption"]
IMG_KEYS = ["image", "img", "picture"]


class CRNN(nn.Module):
    """Identical to quickstart*.py -- same keys, so checkpoints load either way."""
    def __init__(s, n_cls):
        super().__init__()
        def blk(i, o, p):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.MaxPool2d(p))
        s.cnn = nn.Sequential(blk(1, 64, (2, 2)), blk(64, 128, (2, 2)),
                              blk(128, 256, (2, 1)), blk(256, 512, (2, 1)),
                              nn.Conv2d(512, 512, (4, 1)), nn.BatchNorm2d(512), nn.ReLU(True))
        s.rnn = nn.LSTM(512, 256, 2, bidirectional=True, batch_first=True, dropout=0.1)
        s.fc = nn.Linear(512, n_cls)
    def forward(s, x):
        return s.fc(s.rnn(s.cnn(x).squeeze(2).permute(0, 2, 1))[0])


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prep(img):
    img = img.convert("L")
    w, h = img.size
    nw = max(8, min(W, int(w * H / max(h, 1))))
    img = img.resize((nw, H), Image.BILINEAR)
    c = Image.new("L", (W, H), 255)
    c.paste(img, (0, 0))
    return c


def to_tensor(img, aug=False):
    a = np.array(img, dtype=np.float32)
    if aug:
        if random.random() < 0.4: a += np.random.normal(0, random.uniform(3, 12), a.shape)
        if random.random() < 0.4: a = (a - 128) * random.uniform(0.7, 1.3) + 128
    return torch.from_numpy((np.clip(a, 0, 255) / 255.0 - 0.5) / 0.5)[None]


def row_to_sample(row, stoi, max_len, khmer_only):
    tk = next((k for k in TEXT_KEYS if k in row), None)
    ik = next((k for k in IMG_KEYS if k in row), None)
    if not tk or not ik: return None
    lbl = normalize_khmer(row[tk])
    if not lbl or len(lbl) > max_len: return None
    if khmer_only and khmer_ratio(lbl) < 0.8: return None
    ids = [stoi[c] for c in lbl if c in stoi] if stoi else None
    if stoi is not None and not ids: return None
    return row[ik], lbl, ids


class MixStream(IterableDataset):
    def __init__(s, repos, weights, stoi, max_len, khmer_only, buf, seed, skip):
        s.repos, s.weights, s.stoi = repos, weights, stoi
        s.max_len, s.khmer_only, s.buf, s.seed, s.skip = max_len, khmer_only, buf, seed, skip

    def __iter__(s):
        from datasets import load_dataset
        rng = random.Random(s.seed)
        its, ws = [], []
        for repo, w in zip(s.repos, s.weights):
            try:
                ds = load_dataset(repo, split="train", streaming=True)
                if s.buf: ds = ds.shuffle(seed=s.seed, buffer_size=s.buf)
                its.append(iter(ds)); ws.append(w)
            except Exception as e:
                print(f"  [skip] {repo}: {e}", flush=True)
        if not its: return
        skipped = 0
        while its:
            i = rng.choices(range(len(its)), weights=ws)[0]
            try:
                row = next(its[i])
            except StopIteration:
                its.pop(i); ws.pop(i); continue
            except Exception:
                continue
            if skipped < s.skip:
                skipped += 1; continue
            out = row_to_sample(row, s.stoi, s.max_len, s.khmer_only)
            if out is None: continue
            img, lbl, ids = out
            try:
                t = to_tensor(prep(img), aug=True)
            except Exception:
                continue
            yield t, torch.tensor(ids), lbl


def collate(b):
    im, tg, tx = zip(*b)
    return torch.stack(im), torch.cat(tg), torch.tensor([len(t) for t in tg]), list(tx)


def edit(a, b):
    if a == b: return 0
    if not a: return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_charset(repos, weights, n, max_len, khmer_only):
    """Only used when there is no checkpoint to inherit a charset from."""
    from collections import Counter
    print(f"building charset from {n} samples ...")
    ds = MixStream(repos, weights, None, max_len, khmer_only, 2000, 0, 0)
    cnt, seen = Counter(), 0
    from datasets import load_dataset
    for repo, w in zip(repos, weights):
        try:
            d = load_dataset(repo, split="train", streaming=True)
        except Exception:
            continue
        got = 0
        for row in d:
            out = row_to_sample(row, None, max_len, khmer_only)
            if out is None: continue
            cnt.update(out[1]); got += 1; seen += 1
            if got >= n // len(repos): break
    chars = sorted(c for c, k in cnt.items() if k >= 3)
    print(f"  {seen} samples -> {len(chars)} classes")
    return chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--khmer-only", action="store_true")
    ap.add_argument("--shuffle-buffer", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init", default="", help="checkpoint for weights+charset (optional)")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "mac_state.pt"))
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()

    weights = args.weights or (DEFAULT_WEIGHTS if args.repos == DEFAULT_REPOS
                               else [1.0] * len(args.repos))
    assert len(weights) == len(args.repos)
    assert 2 * args.max_len <= W // 4

    dev = pick_device(args.device)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    best_path = str(out) + ".best.pt"
    print(f"device={dev}  (fp32: MPS has no usable fp16 autocast)")
    print(f"batch={args.batch_size}  workers={args.workers}")

    resuming = out.exists()
    if resuming:
        ck = torch.load(out, map_location="cpu", weights_only=False)
        chars = ck["chars"]; start_step = ck.get("step", 0)
        best = ck.get("best", 9.9); rc = ck.get("resume_count", 0) + 1
        print(f"RESUMING {out}: step {start_step}/{args.steps} best {best:.4f}")
    elif args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        chars = ck["chars"]; start_step, best, rc = 0, 9.9, 0
        print(f"INIT from {args.init}: charset {len(chars)}, prior CER {ck.get('cer')}")
    else:
        ck = None
        chars = build_charset(args.repos, weights, 4000, args.max_len, args.khmer_only)
        start_step, best, rc = 0, 9.9, 0

    seed = 1234 + rc * 7919
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    print(f"charset={len(chars)}  seed={seed}")
    if start_step >= args.steps:
        print("target reached; raise --steps."); return

    print(f"building eval set ({args.val_size}) ...")
    from datasets import load_dataset
    val = []
    per = max(1, args.val_size // len(args.repos))
    for repo in args.repos:
        try:
            d = load_dataset(repo, split="train", streaming=True)
        except Exception as e:
            print(f"  [skip] {repo}: {e}"); continue
        got = 0
        for row in d:
            o = row_to_sample(row, stoi, args.max_len, args.khmer_only)
            if o is None: continue
            try: val.append((to_tensor(prep(o[0])), o[1]))
            except Exception: continue
            got += 1
            if got >= per: break
        print(f"  {got:4d}  {repo}")
    print(f"  total {len(val)}")

    tl = DataLoader(MixStream(args.repos, weights, stoi, args.max_len,
                              args.khmer_only, args.shuffle_buffer, seed, per),
                    batch_size=args.batch_size, collate_fn=collate,
                    num_workers=args.workers)

    model = CRNN(len(chars) + 1).to(dev)
    if ck is not None and "model" in ck:
        try:
            model.load_state_dict(ck["model"])
            print("  loaded weights")
        except Exception as e:
            print(f"  WARNING: weights not loaded ({e}); training from scratch")
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                              pct_start=0.05)
    if resuming and "opt" in ck:
        opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sched"])
        print(f"  restored optimizer (lr {sch.get_last_lr()[0]:.2e})")
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
        for i in range(0, len(val), args.batch_size):
            ch = val[i:i + args.batch_size]
            lg = model(torch.stack([c[0] for c in ch]).to(dev))
            P += [decode(s) for s in lg.argmax(-1).cpu().tolist()]
            G += [c[1] for c in ch]
        model.train()
        return (sum(edit(p, g) for p, g in zip(P, G)) / max(sum(len(g) for g in G), 1),
                sum(p != g for p, g in zip(P, G)) / len(G), list(zip(P, G))[:2])

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sch.state_dict(), "chars": chars, "step": step,
                    "best": best, "resume_count": rc}, out)

    stop = {"now": False}
    def on_sig(sig, frm):
        stop["now"] = True
        print("\ninterrupt received -- saving at next step boundary ...")
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    print(f"\ntraining to {args.steps} (from {start_step})\n" + "-" * 60)
    step, run, t0 = start_step, 0.0, time.time()
    while step < args.steps and not stop["now"]:
        for im, tg, ln, tx in tl:
            if step >= args.steps or stop["now"]: break
            im = im.to(dev)
            lg = model(im)
            lp = lg.log_softmax(-1).permute(1, 0, 2).float()
            loss = ctc(lp, tg, torch.full((im.size(0),), lg.size(1), dtype=torch.long), ln)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sch.last_epoch < args.steps - 1: sch.step()
            run += loss.item(); step += 1
            if step % 50 == 0:
                ips = 50 * args.batch_size / (time.time() - t0)
                eta = (args.steps - step) * args.batch_size / max(ips, 1) / 3600
                print(f"step {step:6d}/{args.steps}  loss {run/50:7.4f}  "
                      f"{ips:5.1f} img/s  eta {eta:.1f}h", flush=True)
                run, t0 = 0.0, time.time()
            if step % args.save_every == 0:
                save(step)
            if step % args.eval_every == 0 or step == args.steps:
                cer, wer, ex = ev()
                flag = ""
                if cer < best:
                    best = cer; flag = "  <- best"
                    torch.save({"model": model.state_dict(), "chars": chars,
                                "cer": cer, "step": step}, best_path)
                save(step)
                print(f"  >> CER {cer:.4f}  WER {wer:.4f}{flag}")
                for p, g in ex[:1]:
                    print(f"     pred: {p}\n     gold: {g}")
                t0 = time.time()

    save(step)
    print(f"\nstopped at step {step}. best CER {best:.4f}")
    print(f"state -> {out}\nbest  -> {best_path}")
    print("run the same command again to continue.")


if __name__ == "__main__":
    main()
