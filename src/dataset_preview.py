"""Show what the data actually looks like: image next to its transcript.

Before trusting any CER number it is worth seeing the raw samples -- how long
the lines are, how clean the scans are, whether the transcripts match what is
legible. This renders samples straight from a .pt bundle with no model
involved, so what you see is exactly what the loader feeds the network.

    python -u src/dataset_preview.py --set results/test_set.pt
"""
import argparse, base64, html, io, sys
from pathlib import Path
from collections import Counter

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from seeds import set_all_seeds
import random

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["printed_B", "printed_C", "printed_A", "camera", "handwriting"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(ROOT / "results" / "test_set.pt"))
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "dataset_preview.html"))
    a = ap.parse_args()

    set_all_seeds()
    rng = random.Random(42)
    bundle = torch.load(a.set, map_location="cpu", weights_only=False)
    items = bundle["items"] if "items" in bundle else bundle
    print(f"{a.set}: {len(items)} samples")

    by_dom = {}
    for it in items:
        by_dom.setdefault(it.get("domain", "all"), []).append(it)

    doms = [d for d in ORDER if d in by_dom] + \
           [d for d in by_dom if d not in ORDER]

    rows = []
    for dom in doms:
        pool = by_dom[dom]
        picks = rng.sample(pool, min(a.per_domain, len(pool)))
        lens = [len(it["text"]) for it in pool]
        sizes = [Image.open(io.BytesIO(it["jpeg"])).size for it in picks]
        w = sum(s[0] for s in sizes) / len(sizes)
        h = sum(s[1] for s in sizes) / len(sizes)
        seen = pool[0].get("seen_in_training")
        tag = ("seen in training" if seen else "UNSEEN in training") \
              if seen is not None else ""
        print(f"  {dom:12s} n={len(pool):5d}  chars {min(lens)}-{max(lens)} "
              f"(mean {sum(lens)/len(lens):.0f})  ~{w:.0f}x{h:.0f}px")
        rows.append(f'''<h2>{dom} <small>{tag} &middot; n={len(pool)} &middot;
 {min(lens)}&ndash;{max(lens)} chars (mean {sum(lens)/len(lens):.0f}) &middot;
 ~{w:.0f}&times;{h:.0f}px</small></h2>''')
        for it in picks:
            b64 = base64.b64encode(it["jpeg"]).decode()
            rows.append(f'''<div class="card">
  <img src="data:image/jpeg;base64,{b64}">
  <div class="txt">
    <div class="line"><span class="lbl">text</span><span class="khm">{html.escape(it["text"])}</span></div>
    <div class="line"><span class="lbl">len</span><b>{len(it["text"])}</b> chars</div>
  </div>
</div>''')

    doc = f'''<!doctype html><meta charset="utf-8">
<title>Dataset Preview</title>
<style>
 :root {{ --bg:#fff; --fg:#111; --mut:#666; --line:#e5e5e5; --acc:#2563eb; }}
 @media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
    --bg:#15161a; --fg:#eceef2; --mut:#9aa0aa; --line:#2c2f36; --acc:#60a5fa; }} }}
 body {{ background:var(--bg); color:var(--fg); margin:0; padding:24px 16px;
         font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }}
 .wrap {{ max-width:900px; margin:0 auto; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 h2 {{ font-size:16px; margin:32px 0 8px; border-top:1px solid var(--line);
       padding-top:16px; }}
 h2 small, .sub {{ font-weight:400; color:var(--mut); font-size:12px; }}
 .card {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap;
          border:1px solid var(--line); border-left:3px solid var(--acc);
          border-radius:6px; padding:10px; margin-bottom:8px; }}
 .card img {{ max-width:100%; height:auto; background:#fff; border-radius:3px; }}
 .txt {{ flex:1 1 320px; min-width:0; }}
 .line {{ display:flex; gap:8px; align-items:baseline; }}
 .lbl {{ width:36px; flex:none; color:var(--mut); font-size:11px;
         text-transform:uppercase; letter-spacing:.05em; }}
 .khm {{ font-size:17px; word-break:break-word; }}
</style>
<div class="wrap">
<h1>Dataset preview</h1>
<div class="sub">{Path(a.set).name} &middot; {len(items)} samples &middot;
 {a.per_domain} random per domain (seed 42) &middot; no model involved</div>
{''.join(rows)}
</div>'''

    Path(a.out).write_text(doc, encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
