"""Phase 0 -- data audit. CPU only, no GPU needed.

WHY: most from-scratch OCR failures are label bugs, not model bugs. This finds
them in minutes instead of after a 10-hour training run. Run it before packing.

Outputs (written to --out):
  charset.json   the final character vocabulary the model will predict
  stats.json     per-source counts, label lengths, image sizes
  suspects.csv   labels using rare codepoints -- usually encoding errors
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from normalize import normalize_khmer

# label/image columns differ per source; probe in this order
TEXT_KEYS = ["text", "label", "sentence", "transcription", "caption", "ground_truth"]
IMAGE_KEYS = ["image", "img", "picture"]


def find_key(features, candidates):
    for k in candidates:
        if k in features:
            return k
    return None


def audit_source(repo, split="train", limit=None):
    """Return (labels, image_sizes, meta) for one HF dataset."""
    ds = load_dataset(repo, split=split, streaming=True)
    feats = ds.features or {}
    tkey = find_key(feats, TEXT_KEYS)
    ikey = find_key(feats, IMAGE_KEYS)
    if tkey is None:
        # streaming may not expose features; fall back to first row's keys
        first = next(iter(ds))
        tkey = find_key(first, TEXT_KEYS)
        ikey = find_key(first, IMAGE_KEYS)
        if tkey is None:
            raise KeyError(f"{repo}: no text column found; keys={list(first)}")

    labels, sizes = [], []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        labels.append(row[tkey])
        if ikey and hasattr(row.get(ikey), "size"):
            sizes.append(row[ikey].size)   # (w, h)
    return labels, sizes, {"text_key": tkey, "image_key": ikey}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="HF dataset repo ids")
    ap.add_argument("--limit", type=int, default=5000,
                    help="rows sampled per source (audit is a sample, not a full pass)")
    ap.add_argument("--out", default="audit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw_chars, norm_chars = Counter(), Counter()
    stats, suspects = {}, []

    for repo in args.sources:
        print(f"\n=== {repo}")
        try:
            labels, sizes, meta = audit_source(repo, limit=args.limit)
        except Exception as e:
            print(f"  SKIP: {e}")
            stats[repo] = {"error": str(e)}
            continue

        normed = [normalize_khmer(t) for t in labels]
        for t in labels:
            raw_chars.update(t)
        for t in normed:
            norm_chars.update(t)

        lens = [len(t) for t in normed]
        widths = [s[0] for s in sizes] or [0]
        heights = [s[1] for s in sizes] or [0]
        changed = sum(1 for a, b in zip(labels, normed) if a != b)

        stats[repo] = {
            **meta,
            "sampled": len(labels),
            "normalization_changed_pct": round(100 * changed / max(len(labels), 1), 1),
            "label_len": {"min": min(lens), "mean": round(sum(lens) / len(lens), 1),
                          "max": max(lens)},
            "image_w": {"min": min(widths), "mean": round(sum(widths) / len(widths)),
                        "max": max(widths)},
            "image_h": {"min": min(heights), "mean": round(sum(heights) / len(heights)),
                        "max": max(heights)},
        }
        print(f"  {len(labels)} rows | labels changed by normalization: "
              f"{stats[repo]['normalization_changed_pct']}% | "
              f"len {stats[repo]['label_len']} | h {stats[repo]['image_h']}")

    # rare codepoints are the tell for encoding errors
    rare = {c for c, n in norm_chars.items() if n < 20}
    charset = sorted(c for c, n in norm_chars.items() if n >= 20)

    (out / "charset.json").write_text(json.dumps(
        {"charset": charset, "size": len(charset)}, ensure_ascii=False, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    with (out / "suspects.csv").open("w") as f:
        f.write("codepoint,char,name,count\n")
        for c in sorted(rare):
            import unicodedata
            name = unicodedata.name(c, "?")
            f.write(f"U+{ord(c):04X},{c!r},{name},{norm_chars[c]}\n")

    print(f"\n{'='*50}")
    print(f"raw charset:        {len(raw_chars)}")
    print(f"normalized charset: {len(norm_chars)}  ({len(raw_chars) - len(norm_chars)} collapsed)")
    print(f"kept (count>=20):   {len(charset)}")
    print(f"rare/suspect:       {len(rare)}  -> see suspects.csv")
    print(f"\nGATE: expect 80-140 kept. If far above, normalization is incomplete.")
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
