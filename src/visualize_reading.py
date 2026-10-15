"""Show what the model receives and how it reads it, timestep by timestep.

A CRNN does not see a picture -- it sees 128 vertical slices of a 512x64
tensor, and emits one symbol per slice. CTC then collapses repeats and drops
blanks to produce the final string. Rendering those 128 emissions next to the
image makes the mechanism visible: you can see where each character is picked
up, and where the model goes blank on text it cannot resolve.

    python -u src/visualize_reading.py --expansion 3.0
"""
import argparse, base64, html, io, random, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_mac import CRNN, prep, edit, pick_device
from expand_canvas import expand

ROOT = Path(__file__).resolve().parent.parent


def b64(im, maxw=None):
    if maxw and im.size[0] > maxw:
        im = im.resize((maxw, int(im.size[1] * maxw / im.size[0])))
    buf = io.BytesIO(); im.convert("L").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def read_with_steps(model, itos, img, dev):
    """Return (decoded, per-timestep symbols, model input image)."""
    inp = prep(img)
    x = torch.from_numpy((np.array(inp, dtype=np.float32) / 255. - .5) / .5)[None][None].to(dev)
    logits = model(x)[0]                       # (T, C)
    ids = logits.argmax(-1).cpu().tolist()     # T predictions
    steps = [("" if i == 0 else itos.get(i, "?")) for i in ids]

    out, prev = [], None
    for i in ids:
        if i != 0 and i != prev:
            out.append(itos.get(i, ""))
        prev = i
    return "".join(out), steps, inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/A2_crnn_transfer.pt")
    ap.add_argument("--expansion", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "how_it_reads.html"))
    a = ap.parse_args()

    dev = pick_device("auto")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    itos = {i + 1: c for i, c in enumerate(chars)}
    model = CRNN(len(chars) + 1).to(dev).eval()
    model.load_state_dict(ck["model"])

    bundle = torch.load(ROOT / "results" / "train_subset.pt", map_location="cpu",
                        weights_only=False)
    pairs = bundle["val"][:a.n]

    blocks = []
    for n, (jb, gold) in enumerate(pairs, 1):
        before = Image.open(io.BytesIO(jb)).convert("L")
        after, _, eff = expand(before, a.expansion, "portrait", random.Random(42 + n))

        rows = []
        for label, img in (("BEFORE", before), (f"AFTER {a.expansion:g}×", after)):
            pred, steps, inp = read_with_steps(model, itos, img, dev)
            cer = edit(pred, gold) / max(len(gold), 1)
            emitted = sum(1 for s in steps if s)

            cells = "".join(
                f'<span class="{"c" if s else "b"}">{html.escape(s) if s else "·"}</span>'
                for s in steps)

            rows.append(f'''<div class="side">
  <div class="tag">{label} &middot; {img.size[0]}×{img.size[1]}px &middot;
    aspect {img.size[0]/img.size[1]:.2f}
    {"" if label=="BEFORE" else f"&middot; text ≈{eff:.0f}px"}</div>
  <img class="orig" src="data:image/png;base64,{b64(img, 420)}">
  <div class="cap">what the model receives &mdash; 512×64</div>
  <img class="inp" src="data:image/png;base64,{b64(inp)}">
  <div class="cap">128 timesteps &mdash; {emitted} emit a character, the rest are CTC blanks</div>
  <div class="steps">{cells}</div>
  <div class="res"><span class="lbl">reads</span>
    <span class="khm">{html.escape(pred) or "<em>nothing</em>"}</span></div>
  <div class="res"><span class="lbl">CER</span><b class="{"ok" if cer<0.05 else "bad"}">{cer*100:.1f}%</b></div>
</div>''')

        blocks.append(f'''<div class="card">
  <div class="gold"><span class="lbl">truth</span><span class="khm">{html.escape(gold)}</span></div>
  <div class="pair">{"".join(rows)}</div>
</div>''')

    doc = f'''<!doctype html><meta charset="utf-8"><title>How it reads</title><style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#e5e5e5;--acc:#2563eb;--ok:#0a7d33;--bad:#c2281d;--blank:#c9ccd1}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#15161a;--fg:#eceef2;--mut:#9aa0aa;--line:#2c2f36;--acc:#60a5fa;--ok:#4ade80;--bad:#f87171;--blank:#3a3f47}}}}
body{{background:var(--bg);color:var(--fg);margin:0;padding:24px 16px;font:14px/1.5 ui-sans-serif,system-ui,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto}}h1{{font-size:20px;margin:0 0 4px}}
.sub{{color:var(--mut);font-size:12px;margin-bottom:22px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:16px}}
.gold{{display:flex;gap:8px;align-items:baseline;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid var(--line)}}
.pair{{display:flex;gap:18px;flex-wrap:wrap}}
.side{{flex:1 1 420px;min-width:0;border-left:3px solid var(--acc);padding-left:12px}}
.tag{{font-size:12px;font-weight:600;margin-bottom:8px}}
.cap{{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin:10px 0 4px}}
img{{background:#fff;border:1px solid var(--line);border-radius:3px;display:block;max-width:100%}}
img.inp{{width:100%;image-rendering:pixelated}}
.steps{{display:flex;flex-wrap:wrap;gap:1px;font-size:13px;line-height:1.25;
  background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:6px}}
.steps span{{display:inline-block;min-width:11px;text-align:center}}
.steps .b{{color:var(--blank)}}
.steps .c{{color:var(--acc);font-weight:600}}
.res{{display:flex;gap:8px;align-items:baseline;margin-top:8px}}
.lbl{{width:42px;flex:none;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
.khm{{font-size:16px;word-break:break-word}}
b.ok{{color:var(--ok)}} b.bad{{color:var(--bad)}}
</style><div class="wrap">
<h1>How the model reads an image</h1>
<div class="sub">Every input is squashed to 512×64, then sliced into 128 vertical timesteps.
 The model emits one symbol per slice; CTC drops the blanks (·) and collapses repeats.
 Left is the original crop, right is the same line after {a.expansion:g}× canvas expansion.</div>
{"".join(blocks)}</div>'''

    Path(a.out).write_text(doc, encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
