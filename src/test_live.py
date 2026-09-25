import io, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/Users/seouldorito/khmer-ocr/src")
from train_mac import CRNN, prep, edit, pick_device

ck_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/A2_crnn_transfer.pt"
dom_filter = sys.argv[2] if len(sys.argv) > 2 else None
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12

dev = pick_device("auto")
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
chars = ck["chars"]; itos = {i+1: c for i, c in enumerate(chars)}
m = CRNN(len(chars)+1).to(dev).eval(); m.load_state_dict(ck["model"])
ts = torch.load("results/test_set.pt", map_location="cpu", weights_only=False)
items = [it for it in ts["items"] if not dom_filter or it["domain"] == dom_filter][:N]

print(f"model: {Path(ck_path).name}   device: {dev}   charset: {len(chars)}")
print(f"testing {len(items)} images" + (f" from {dom_filter}" if dom_filter else "") + "\n")

tot_e = tot_c = ok = 0
for n, it in enumerate(items, 1):
    img = Image.open(io.BytesIO(it["jpeg"]))
    x = torch.from_numpy((np.array(prep(img), dtype=np.float32)/255.0 - 0.5)/0.5)[None][None].to(dev)
    t0 = time.time()
    with torch.no_grad():
        s = m(x).argmax(-1)[0].cpu().tolist()
    ms = (time.time()-t0)*1000
    o, p = [], None
    for t in s:
        if t != 0 and t != p: o.append(itos.get(t, ""))
        p = t
    pred = "".join(o); gold = it["text"]
    e = edit(pred, gold); c = max(len(gold), 1)
    tot_e += e; tot_c += c; ok += pred == gold
    mark = "OK  " if pred == gold else ("near" if e/c < .2 else "FAIL")
    print(f"[{n:2d}/{len(items)}] {it['domain']:11s} {img.size[0]:4d}x{img.size[1]:<3d} {ms:5.0f}ms  {mark}  CER {e/c*100:5.1f}%")
    print(f"        gold: {gold}")
    print(f"        pred: {pred or '(empty)'}")
    if pred != gold:
        print(f"        diff: {e} edit(s) over {c} chars")
    print()

print(f"{'='*60}\n{ok}/{len(items)} exact   overall CER {tot_e/tot_c*100:.2f}%")
