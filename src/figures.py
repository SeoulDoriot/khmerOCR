"""Generate the comparison figures the rubric requires.

  fig1_learning_curves.png  train vs val loss and val CER per approach, which
                            is where under/overfitting is visible
  fig2_comparison.png       final CER per approach overall and per domain,
                            separating domains seen in training from unseen

Both are written to results/ and referenced from the README.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

APPROACHES = [
    ("A1_crnn_scratch",  "A1  CRNN+CTC, scratch",  "#2a78d6"),
    ("A2_crnn_transfer", "A2  CRNN+CTC, transfer", "#1baf7a"),
    ("A3_attn_scratch",  "A3  Attention, scratch", "#eb6834"),
]


def read_curve(tag):
    p = RES / f"curve_{tag}.csv"
    if not p.exists():
        return None
    import csv
    rows = list(csv.DictReader(p.open()))
    return {k: [float(r[k]) for r in rows] for k in
            ("step", "train_loss", "val_loss", "val_cer", "val_wer")}


def fig_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for tag, label, colour in APPROACHES:
        c = read_curve(tag)
        if not c:
            continue
        axes[0].plot(c["step"], c["train_loss"], color=colour, label=label, lw=1.8)
        axes[1].plot(c["step"], c["val_loss"], color=colour, label=label, lw=1.8)
        axes[2].plot(c["step"], [v * 100 for v in c["val_cer"]], color=colour,
                     label=label, lw=1.8, marker="o", ms=3)
    for ax, title, ylab in zip(
            axes,
            ["Training loss", "Validation loss", "Validation CER"],
            ["loss", "loss", "CER (%)"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Learning curves — identical data, seed and schedule for all approaches",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = RES / "fig1_learning_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def fig_comparison():
    ev = json.loads((RES / "evaluation.json").read_text())["results"]
    tags = [t for t, _, _ in APPROACHES if t in ev]
    domains = list(ev[tags[0]]["domains"].keys())
    seen = {d: ev[tags[0]]["domains"][d]["seen_in_training"] for d in domains}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6),
                             gridspec_kw={"width_ratios": [1, 2.1]})

    # left: overall CER
    vals = [ev[t]["overall"]["cer"] * 100 for t in tags]
    cols = [c for t, _, c in APPROACHES if t in ev]
    bars = axes[0].bar(range(len(tags)), vals, color=cols, width=0.6)
    axes[0].set_xticks(range(len(tags)))
    axes[0].set_xticklabels([t.split("_")[0] for t in tags])
    axes[0].set_ylabel("CER (%)")
    axes[0].set_title("Overall CER on the frozen test set\n(1,840 samples, 5 domains)",
                      fontsize=11)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                     ha="center", fontsize=10)
    axes[0].grid(axis="y", alpha=0.25, lw=0.6)
    axes[0].spines[["top", "right"]].set_visible(False)

    # right: per-domain
    w = 0.26
    for i, (t, label, colour) in enumerate([a for a in APPROACHES if a[0] in ev]):
        xs = [x + (i - 1) * w for x in range(len(domains))]
        ys = [ev[t]["domains"][d]["cer"] * 100 for d in domains]
        axes[1].bar(xs, ys, width=w, color=colour, label=label)
    axes[1].set_xticks(range(len(domains)))
    axes[1].set_xticklabels([f"{d}\n{'seen' if seen[d] else 'UNSEEN'}" for d in domains],
                            fontsize=9)
    axes[1].set_ylabel("CER (%)")
    axes[1].set_title("CER by domain — three of five were never trained on",
                      fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25, lw=0.6)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = RES / "fig2_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_curves()
    fig_comparison()
