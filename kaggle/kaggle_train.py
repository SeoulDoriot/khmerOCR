"""Khmer OCR training on Kaggle. Continues the Colab checkpoint.

KAGGLE SPECIFICS:
  * No Google Drive. The starting checkpoint arrives as a Kaggle Dataset
    mounted read-only at /kaggle/input/<slug>/. Output goes to
    /kaggle/working/, which persists after the session ends.
  * "Save & Run All" executes in the background with the browser closed --
    this is why Kaggle suits long runs better than free Colab.
  * Internet must be enabled in notebook settings or HF streaming fails.

RESUME ORDER (so re-running the notebook always does the right thing):
  1. /kaggle/working/kaggle_state.pt  -- this run's own progress, if any
  2. /kaggle/input/**/quickstart2_crnn.pt -- the Colab model, first time only
Charset ALWAYS comes from the checkpoint, never rebuilt from data: a rebuilt
charset would change the output layer and silently discard the trained model.
"""
import argparse, glob, os, random, signal, time, unicodedata
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset

COENG = "្"
INHERENT = {"឴", "឵"}
ZW = {"​", "‌", "‍", "﻿", "­"}
_cons = lambda c: "ក" <= c <= "អ"
_vow = lambda c: "ា" <= c <= "ៅ"
_shift = lambda c: c in ("៉", "៊")
_kh = lambda c: "ក" <= c <= "៿"

def _clusters(t):
    out, cur = [], ""
    for ch in t:
        new = _cons(ch) or ("ឣ" <= ch <= "ឳ")
        if new and cur.endswith(COENG): new = False
        if new and cur: out.append(cur); cur = ch
        elif new: cur = ch
        else:
            if not cur: out.append(ch); continue
            cur += ch
    if cur: out.append(cur)
    return out

def _canon(cl):
    if not cl: return cl
    base, i = cl[0], 1
    co, sh, vo, sg = [], [], [], []
    while i < len(cl):
        ch = cl[i]
        if ch == COENG and i + 1 < len(cl):
            co.append(COENG + cl[i + 1]); i += 2; continue
        (sh if _shift(ch) else vo if _vow(ch) else sg).append(ch); i += 1
    co.sort(); sg.sort()
    return base + "".join(co) + "".join(sh) + "".join(vo) + "".join(sg)

def normalize_khmer(t):
    if not t: return ""
    t = unicodedata.normalize("NFC", t)
    t = "".join(c for c in t if c not in ZW and c not in INHERENT)
    return " ".join("".join(_canon(c) for c in _clusters(t)).split())

def khmer_ratio(t):
    L = [c for c in t if not c.isspace()]
    return sum(_kh(c) for c in L) / len(L) if L else 0.0


class CRNN(nn.Module):
    """Must stay identical to quickstart2/5 or the checkpoint will not load."""
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


H, W = 64, 512
TEXT_KEYS = ["text", "label", "sentence", "transcription", "ground_truth", "caption"]
IMG_KEYS = ["image", "img", "picture"]
REPOS = [
    "Chanrith123333/khmer_english_ocr_image_line",
    "SoyVitou/KhmerSynthetic1M",
    "KimkosalYon/khmer-ocr-200k-siemreap-arial",
    "SoyVitou/62k-images-khmer-printed-dataset",
]
WEIGHTS = [0.35, 0.30, 0.15, 0.20]

def prep(img):
    img = img.convert("L")
    w, h = img.size
    nw = max(8, min(W, int(w * H / max(h, 1))))
    img = img.resize((nw, H), Image.BILINEAR)
    c = Image.new("L", (W, H), 255); c.paste(img, (0, 0)); return c

def to_tensor(img, aug=False):
    a = np.array(img, dtype=np.float32)
    if aug:
        if random.random() < 0.4: a += np.random.normal(0, random.uniform(3, 12), a.shape)
        if random.random() < 0.4: a = (a - 128) * random.uniform(0.7, 1.3) + 128
    return torch.from_numpy((np.clip(a, 0, 255) / 255.0 - 0.5) / 0.5)[None]

def row_to(row, stoi, max_len, khmer_only):
    tk = next((k for k in TEXT_KEYS if k in row), None)
    ik = next((k for k in IMG_KEYS if k in row), None)
    if not tk or not ik: return None
    lbl = normalize_khmer(row[tk])
    if not lbl or len(lbl) > max_len: return None
    if khmer_only and khmer_ratio(lbl) < 0.8: return None
    ids = [stoi[c] for c in lbl if c in stoi]
    if not ids: return None
    return row[ik], lbl, ids


class MixStream(IterableDataset):
    def __init__(s, repos, weights, stoi, max_len, khmer_only, buf, seed, skip):
        s.r, s.w, s.stoi = repos, weights, stoi
        s.max_len, s.ko, s.buf, s.seed, s.skip = max_len, khmer_only, buf, seed, skip
    def __iter__(s):
        from datasets import load_dataset
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        rng = random.Random(s.seed * 1000 + wid)
        its, ws = [], []
        for repo, w in zip(s.r, s.w):
            try:
                d = load_dataset(repo, split="train", streaming=True)
                if s.buf: d = d.shuffle(seed=s.seed + wid, buffer_size=s.buf)
                if nw > 1: d = d.shard(num_shards=nw, index=wid)
                its.append(iter(d)); ws.append(w)
            except Exception as e:
                if wid == 0: print(f"  [skip] {repo}: {e}", flush=True)
        if not its: return
        skipped = 0
        while its:
            i = rng.choices(range(len(its)), weights=ws)[0]
            try: row = next(its[i])
            except StopIteration: its.pop(i); ws.pop(i); continue
            except Exception: continue
            if skipped < s.skip: skipped += 1; continue
            o = row_to(row, s.stoi, s.max_len, s.ko)
            if o is None: continue
            try: t = to_tensor(prep(o[0]), aug=True)
            except Exception: continue
            yield t, torch.tensor(o[2]), o[1]

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


def find_init():
    """The Colab checkpoint, wherever Kaggle mounted the dataset."""
    for pat in ["/kaggle/input/*/quickstart2_crnn.pt",
                "/kaggle/input/*/*/quickstart2_crnn.pt",
                "/kaggle/input/*/*.pt"]:
        hits = sorted(glob.glob(pat))
        if hits: return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--khmer-only", action="store_true", default=True)
    ap.add_argument("--shuffle-buffer", type=int, default=1000)
    ap.add_argument("--val-per-repo", type=int, default=250)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="/kaggle/working/kaggle_state.pt")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--init", default="", help="explicit path to the starting checkpoint")
    ap.add_argument("--max-hours", type=float, default=11.0,
                    help="stop and save before Kaggle's 12h session cap")
    a = ap.parse_args()

    assert 2 * a.max_len <= W // 4
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
    if dev.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: no GPU. Enable it in notebook settings (Accelerator).")

    best_path = a.out + ".best.pt"
    resuming = os.path.exists(a.out)
    src = a.out if resuming else (a.init if a.init and os.path.exists(a.init) else find_init())
    if src is None:
        raise SystemExit("No checkpoint found. Attach your Kaggle Dataset containing "
                         "quickstart2_crnn.pt via 'Add Input'.")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    start_step = ck.get("step", 0) if resuming else 0
    best = ck.get("best", 9.9) if resuming else 9.9
    rc = ck.get("resume_count", 0) + 1 if resuming else 0
    seed = 1234 + rc * 7919
    print(f"{'RESUME' if resuming else 'CONTINUE FROM COLAB'}: {src}")
    print(f"  charset={len(chars)}  step={start_step}/{a.steps}  seed={seed}"
          + (f"  best={best:.4f}" if resuming else f"  prior CER={ck.get('cer')}"))
    if start_step >= a.steps:
        print("target reached; raise --steps."); return

    print(f"\nbuilding eval set ({a.val_per_repo}/corpus) ...", flush=True)
    from datasets import load_dataset
    val = []
    for repo in REPOS:
        try: d = load_dataset(repo, split="train", streaming=True)
        except Exception as e:
            print(f"  [skip] {repo}: {e}"); continue
        got = 0
        for row in d:
            o = row_to(row, stoi, a.max_len, a.khmer_only)
            if o is None: continue
            try: val.append((to_tensor(prep(o[0])), o[1], repo))
            except Exception: continue
            got += 1
            if got >= a.val_per_repo: break
        print(f"  {got:4d}  {repo}", flush=True)
    print(f"  total {len(val)}")

    tl = DataLoader(MixStream(REPOS, WEIGHTS, stoi, a.max_len, a.khmer_only,
                              a.shuffle_buffer, seed, a.val_per_repo),
                    batch_size=a.batch_size, collate_fn=collate,
                    num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0)

    model = CRNN(len(chars) + 1).to(dev)
    model.load_state_dict(ck["model"])
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M (weights loaded)")
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    if resuming and "opt" in ck:
        opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        print(f"  optimizer restored (lr {sch.get_last_lr()[0]:.2e})")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def decode(seq):
        o, p = [], None
        for i in seq:
            if i != 0 and i != p: o.append(itos.get(i, ""))
            p = i
        return "".join(o)

    @torch.no_grad()
    def ev():
        model.eval(); P, G, R = [], [], []
        for i in range(0, len(val), a.batch_size):
            ch = val[i:i + a.batch_size]
            x = torch.stack([c[0] for c in ch]).to(dev)
            with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
                lg = model(x)
            P += [decode(s) for s in lg.argmax(-1).cpu().tolist()]
            G += [c[1] for c in ch]; R += [c[2] for c in ch]
        model.train()
        cer = sum(edit(p, g) for p, g in zip(P, G)) / max(sum(len(g) for g in G), 1)
        wer = sum(p != g for p, g in zip(P, G)) / len(G)
        by = {}
        for r in set(R):
            idx = [i for i, x in enumerate(R) if x == r]
            by[r] = sum(edit(P[i], G[i]) for i in idx) / max(sum(len(G[i]) for i in idx), 1)
        return cer, wer, by, list(zip(P, G))[:2]

    def save(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sch.state_dict(), "scaler": scaler.state_dict(),
                    "chars": chars, "step": step, "best": best,
                    "resume_count": rc}, a.out)

    if not resuming:
        c0, w0, by0, _ = ev()
        best = c0
        print(f"\nSTARTING POINT on this eval set: CER {c0:.4f}  WER {w0:.4f}")
        for r, c in sorted(by0.items(), key=lambda x: -x[1]):
            print(f"   {c:.4f}  {r.split('/')[-1]}")
        print("  (different eval set than the 0.65% figure -- compare Kaggle to Kaggle)")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    t_start = time.time()

    print(f"\ntraining to {a.steps} (from {start_step})\n" + "-" * 60, flush=True)
    step, run, t0 = start_step, 0.0, time.time()
    while step < a.steps and not stop["now"]:
        for im, tg, ln, tx in tl:
            if step >= a.steps or stop["now"]: break
            if (time.time() - t_start) / 3600 > a.max_hours:
                print(f"\nreached --max-hours {a.max_hours}; saving and stopping cleanly.")
                stop["now"] = True; break
            im = im.to(dev, non_blocking=True)
            with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
                lg = model(im)
            lp = lg.log_softmax(-1).permute(1, 0, 2).float()
            loss = ctc(lp, tg, torch.full((im.size(0),), lg.size(1), dtype=torch.long), ln)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if sch.last_epoch < a.steps - 1: sch.step()
            run += loss.item(); step += 1
            if step % 200 == 0:
                ips = 200 * a.batch_size / (time.time() - t0)
                print(f"step {step:6d}/{a.steps}  loss {run/200:7.4f}  {ips:5.0f} img/s  "
                      f"{(time.time()-t_start)/3600:.1f}h", flush=True)
                run, t0 = 0.0, time.time()
            if step % a.save_every == 0: save(step)
            if step % a.eval_every == 0 or step == a.steps:
                cer, wer, by, ex = ev()
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
    print("\n" + "=" * 60)
    print(f"stopped at step {step}/{a.steps}.  best CER {best:.4f}")
    print(f"resume state : {a.out}")
    print(f"best weights : {best_path}")
    print("Download both from the Output tab, then re-upload the state file as a")
    print("Dataset version to continue in the next session.")


if __name__ == "__main__":
    main()
