"""Qualitative results: show each test image with what the model read off it.

The CER table says a domain is hard; it does not say *how* it fails. This
renders the actual images next to prediction and ground truth so the failure
mode is visible -- whether the model is dropping diacritics, confusing
subscripts, or producing nothing at all.

    python -u src/predict_gallery.py --ckpt checkpoints/A2_crnn_transfer.pt
"""
import argparse, base64, html, io, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from seeds import set_all_seeds
from train_mac import CRNN, prep, edit, pick_device

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["printed_B", "printed_C", "printed_A", "camera", "handwriting"]


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    chars = ck["chars"]
    model = CRNN(len(chars) + 1).to(dev).eval()
    model.load_state_dict(ck["model"])
    return model, chars


@torch.no_grad()
def predict(model, chars, items, dev, bs=48):
    itos = {i + 1: c for i, c in enumerate(chars)}
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = torch.stack([
            torch.from_numpy((np.array(prep(Image.open(io.BytesIO(it["jpeg"]))),
                                       dtype=np.float32) / 255.0 - 0.5) / 0.5)[None]
            for it in chunk]).to(dev)
        for s in model(x).argmax(-1).cpu().tolist():
            o, p = [], None
            for t in s:
                if t != 0 and t != p:
                    o.append(itos.get(t, ""))
                p = t
            out.append("".join(o))
    return out


def cer_of(pred, gold):
    return edit(pred, gold) / max(len(gold), 1)


def img_tag(jpeg):
    b64 = base64.b64encode(jpeg).decode()
    return f'<img src="data:image/jpeg;base64,{b64}">'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/A2_crnn_transfer.pt")
    ap.add_argument("--test-set", default=str(ROOT / "results" / "test_set.pt"))
    ap.add_argument("--per-domain", type=int, default=120,
                    help="how many samples per domain to run inference on")
    ap.add_argument("--show", type=int, default=4,
                    help="best and worst N to display per domain")
    ap.add_argument("--out", default=str(ROOT / "results" / "predictions.html"))
    a = ap.parse_args()

    set_all_seeds()
    dev = pick_device("auto")
    name = Path(a.ckpt).stem
    model, chars = load(a.ckpt, dev)
    print(f"model {name} on {dev}, charset {len(chars)}")

    ts = torch.load(a.test_set, map_location="cpu", weights_only=False)
    items = ts["items"]

    sections = []
    for dom in DOMAINS:
        pool = [it for it in items if it["domain"] == dom][:a.per_domain]
        preds = predict(model, chars, pool, dev)
        scored = sorted(
            ({"it": it, "pred": p, "cer": cer_of(p, it["text"])}
             for it, p in zip(pool, preds)),
            key=lambda r: r["cer"])
        mean = sum(r["cer"] for r in scored) / len(scored)
        exact = sum(r["pred"] == r["it"]["text"] for r in scored)
        seen = pool[0]["seen_in_training"]
        print(f"  {dom:12s} n={len(scored):4d}  CER {mean*100:6.2f}%  exact {exact}/{len(scored)}")
        sections.append((dom, seen, mean, exact, len(scored),
                         scored[:a.show], scored[-a.show:]))

    rows = []
    for dom, seen, mean, exact, n, best, worst in sections:
        tag = "seen in training" if seen else "UNSEEN in training"
        rows.append(f'<h2>{dom} <small>{tag} &middot; n={n} &middot; '
                    f'CER {mean*100:.2f}% &middot; exact {exact}/{n}</small></h2>')
        for label, group in (("best", best), ("worst", worst)):
            rows.append(f'<h3 class="{label}">{label} {len(group)}</h3>')
            for r in group:
                gold, pred, c = r["it"]["text"], r["pred"], r["cer"]
                ok = "ok" if pred == gold else ("near" if c < 0.2 else "bad")
                rows.append(f'''<div class="card {ok}">
  {img_tag(r["it"]["jpeg"])}
  <div class="txt">
    <div class="line"><span class="lbl">gold</span><span class="khm">{html.escape(gold)}</span></div>
    <div class="line"><span class="lbl">pred</span><span class="khm">{html.escape(pred) or "<em>(empty)</em>"}</span></div>
    <div class="line"><span class="lbl">CER</span><b>{c*100:.1f}%</b></div>
  </div>
</div>''')

    doc = f'''<!doctype html><meta charset="utf-8">
<title>Predictions &mdash; {name}</title>
<style>
 :root {{ --bg:#fff; --fg:#111; --mut:#666; --line:#e5e5e5;
          --ok:#0a7d33; --near:#a8760a; --bad:#c2281d; }}
 @media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
    --bg:#15161a; --fg:#eceef2; --mut:#9aa0aa; --line:#2c2f36;
    --ok:#4ade80; --near:#fbbf24; --bad:#f87171; }} }}
 body {{ background:var(--bg); color:var(--fg); margin:0; padding:24px 16px;
         font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }}
 .wrap {{ max-width:900px; margin:0 auto; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 h2 {{ font-size:16px; margin:32px 0 4px; border-top:1px solid var(--line); padding-top:16px; }}
 h2 small, .sub {{ font-weight:400; color:var(--mut); font-size:12px; }}
 h3 {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em;
       color:var(--mut); margin:16px 0 8px; }}
 .card {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap;
          border:1px solid var(--line); border-left-width:3px;
          border-radius:6px; padding:10px; margin-bottom:8px; }}
 .card.ok {{ border-left-color:var(--ok); }}
 .card.near {{ border-left-color:var(--near); }}
 .card.bad {{ border-left-color:var(--bad); }}
 .card img {{ max-width:100%; height:auto; background:#fff; border-radius:3px; }}
 .txt {{ flex:1 1 320px; min-width:0; }}
 .line {{ display:flex; gap:8px; align-items:baseline; }}
 .lbl {{ width:36px; flex:none; color:var(--mut); font-size:11px;
         text-transform:uppercase; letter-spacing:.05em; }}
 .khm {{ font-size:17px; word-break:break-word; }}
</style>
<div class="wrap">
<h1>What the model actually reads</h1>
<div class="sub">{name} &middot; frozen test set, {len(items)} samples &middot;
 best and worst {a.show} shown per domain</div>
{''.join(rows)}
</div>'''

    Path(a.out).write_text(doc, encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
