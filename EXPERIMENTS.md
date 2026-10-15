# Experiment Log

This file records the experiments run *after* the three-approach comparison in
`README.md`. Each section states the question, the measurement, and what the
number actually licenses us to conclude. Negative results are kept.

---

## E1 — Does a low training CER mean the model can read Khmer?

**No.** This is the headline result of the project.

`v7_63800` is a CRNN+CTC model trained for 63,800 steps on a single synthetic
corpus. On its own validation split it reaches **CER 0.0035** — a number that
looks like a solved problem.

Evaluated on a frozen 1,840-line test set spanning five domains:

| domain | n | CER | WER | renderer seen in training |
|---|---|---|---|---|
| printed_A | 300 | **0.0040** | 0.087 | yes |
| printed_B | 300 | 0.6929 | 0.993 | yes |
| printed_C | 200 | 0.4587 | 0.985 | no |
| camera | 200 | 0.7712 | 0.990 | no |
| handwriting | 840 | 0.8033 | 0.973 | no |
| **overall** | **1840** | **0.5214** | **0.835** | — |

Line-level: the model reads **187 / 200** `printed_A` lines perfectly and
**1 / 200** `printed_B` lines perfectly.

The model did not learn the Khmer script. It learned one font renderer. The
0.0035 figure is a measure of how closely the validation split resembles the
training split, not of recognition ability.

**Methodological takeaway:** an in-distribution validation CER is not evidence
of generalization. A held-out *domain* is. This is why the test set is frozen
and split by source rather than by random shuffle.

---

## E2 — Does more training on the same data help?

**No — it makes the model narrower.**

`v8_28800` = `v7_63800` + 28,800 further steps on a joint multi-corpus mix.

| domain | v7 CER | v8 CER | change |
|---|---|---|---|
| printed_A | 0.0040 | 0.0049 | worse |
| printed_B | 0.6929 | 0.6965 | worse |
| printed_C | 0.4587 | 0.4686 | worse |
| camera | 0.7712 | 0.7584 | better |
| handwriting | 0.8033 | 0.8317 | worse |
| **overall** | **0.5214** | **0.5289** | **worse** |

v8 is worse on 5 of 6 measures.

The checkpoint genuinely inherited v7's weights — cosine similarity between
corresponding v7 and v8 parameter tensors is **+0.9901**, versus **−0.0004**
for a randomly initialised control. So this is not a failed load; it is real
continued training that did not help.

**Why:** the run restarted the step counter, which reset the LR scheduler to
its initial high learning rate. A high LR applied to an already-converged
model moves it off its minimum before the shortened run can settle again. The
fix is documented in E3.

---

## E3 — The resume bug

`train.py` resumes from `--out` only if that file already exists. Pointing
`--out` at a *new* filename silently starts from step 0 **and** resets the
cosine LR schedule, even when `--init` loads real weights.

Symptom: loss starts near 6.5 instead of near 0.02.

Fix — copy the checkpoint to the output path before training:

```bash
cp checkpoints/v7_63800.pt /kaggle/working/v9.pt
python src/train.py --init checkpoints/v7_63800.pt --out /kaggle/working/v9.pt ...
```

A second, separate trap: if `--out` reuses a filename that is also an *input*
dataset name, the trained file is not published in the run output at all. One
3h08m run (89,200 steps) was lost this way.

---

## E4 — What actually predicts failure: image height or aspect ratio?

Initial hypothesis: camera images fail because they are low-resolution (short).
**This was wrong, and measurement refuted it.**

Correlation with per-line CER across the full test set:

| feature | correlation with CER |
|---|---|
| image height | −0.017 |
| **aspect ratio (W/H)** | **−0.385** |

Height is uncorrelated. Aspect ratio is the real driver, and the mechanism is
in the preprocessing: `prep()` scales every image to H=64 then right-pads to
W=512. A line whose aspect ratio is far from 8 is either squeezed into a few
timesteps or padded with mostly blank space, and the CTC head sees a different
distribution of glyph widths than it was trained on.

**Consequence:** the fix is not higher-resolution capture. It is either
aspect-ratio-aware bucketing at training time, or augmenting with the aspect
ratios that occur in the wild.

---

## E5 — Small text: expand the canvas, don't shrink the glyphs

To teach the model small text, the glyphs must stay as rendered while the
*canvas* grows around them — shrinking the text instead would just resample it
and destroy the stroke detail we want the model to learn.

`src/expand_canvas.py` pastes the unmodified line onto a larger white canvas at
a jittered position, optionally blurring. After `prep()`, the text occupies
`64 / expansion` pixels of height.

Verified by round-trip (`src/validate_dataset.py`, check 6): the same model on
the same generated images scores

- **69.12% CER** on the full expanded canvas
- **1.04% CER** on the crop taken from the stored bounding box

The 68-point gap proves two things at once: the boxes really do point at the
text, and the degradation is caused by scale, not by a corrupt generator.

Threshold: recognition collapses between **20px and 16px** of text height after
preprocessing.

---

## E6 — Decoding-time fixes

| intervention | effect on residual errors |
|---|---|
| Khmer Unicode normalization (COENG / MUUSIKATOAN canonicalization) | removes ~16% |
| Test-time augmentation (multi-scale, averaged logits) | no improvement |

Normalization is worth applying unconditionally — several "errors" were the
model emitting a valid but differently-encoded form of the correct string. TTA
does not help, which is consistent with E1: the failure is a representation
gap, not decoder noise, and averaging several views of an unreadable input
does not make it readable.

---

## E7 — A data bug worth recording

`src/stream_expand.py` originally wrote four shards that were byte-identical.
Cause: re-entering `for row in ds` restarts a streaming HuggingFace dataset
from row 0, so every shard drew the same rows.

```python
stream = iter(ds)   # ONE iterator for the whole run
while total < target:
    for row in stream:
        ...
```

Caught by hashing the label sets per shard. Any generated dataset should be
checked for duplicate content before it is trained on — four copies of the same
10k rows looks exactly like 40k rows in every log line.

---

## Where this leaves the model

The bottleneck is **data diversity, not capacity**. 9.0M parameters trained to
CER 0.0035 on one renderer is not short of capacity; it is short of variation.
The ranked next steps are in `IMPROVEMENT_PLAN.md`:

1. Joint training across all five corpora with replay, at a *low* LR and a
   correctly-resumed scheduler (E3).
2. Aspect-ratio bucketing instead of fixed W=512 padding (E4).
3. Canvas-expansion augmentation mixed into every corpus (E5).
4. Normalization at decode time (E6) — free, already measured.
