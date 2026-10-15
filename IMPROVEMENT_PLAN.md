# Why the model doesn't understand Khmer — and how to fix it

Written 2026-10-02, after measuring `v7_63800` (63,800 steps, CER 0.35%
in-distribution) on the frozen five-domain test set.

---

## 1. The diagnosis

```
domain        n      CER      exact lines
printed_A    200    0.33%     187/200     ← the font it trained on
printed_C    200   55.69%       3/200
printed_B    200   76.86%       1/200
camera       200   86.66%       2/200
handwriting  200   87.70%       5/200
```

187 perfect lines on one domain, one perfect line on another. **The model
learned a renderer, not a script.** It recognises the exact pixel patterns that
one text-generator produces and has no idea what a Khmer letter is in general.

This was predicted in `METHODOLOGY.md` before any of it was measured:

> A model trained on rendered fonts and evaluated on rendered fonts will report
> near-zero error while being unusable on a photograph.

It is now confirmed with a model that had 63,800 steps of training.

**More steps will not fix it.** Between 39,400 and 63,800 steps, `printed_A`
improved from 0.40% to 0.33% and every other domain stayed flat. The corpus is
the ceiling, not the compute.

**Measured again on 2026-10-15, and it held.** `v8_28800` is `v7_63800` plus
28,800 more single-corpus steps at a reset learning rate. Scored by
`src/evaluate.py` on the frozen test set (1,840 lines; `results/evaluation_v7_v8.json`):

```
domain        n     v7 CER   v8 CER
printed_A    300     0.40%    0.49%   ← worse: the reset LR cost the best number
printed_B    300    69.29%   69.65%
printed_C    200    45.87%   46.86%
camera       200    77.12%   75.84%
handwriting  840    80.33%   83.17%   ← worse
overall     1840    52.14%   52.89%
```

28,800 extra steps bought nothing and the learning-rate reset cost `printed_A`
a quarter of its accuracy. That is the single-corpus ceiling, measured twice.

---

## 2. What is actually wrong, mechanically

**One generator.** Every training run used `--weights 1.0` on a single corpus.
Ten datasets are documented in `DATASETS.md`; one has ever been used for
training.

**No photometric variation.** Training images are clean black-on-white renders.
Real photos have shadows, glare, coloured walls, uneven lighting, inverted
contrast. Tested: inversion, autocontrast, equalise and contrast boost on two
real failure photos — none helped, because the model has no representation of
Khmer robust enough to survive any of it.

**No geometric variation.** Rendered text is perfectly flat and axis-aligned. A
photograph of a wall or a page is never either.

**Handwriting is 0.03% of the data.** 4,200 real handwritten images against
12.1M printed is a 2,900x imbalance. At uniform sampling the model sees
handwriting once every 2,900 batches, which is indistinguishable from never.

---

## 3. The fixes, ranked by impact per hour of work

### Tier 1 — data diversity (the only thing that moves the broken domains)

**1a. Joint multi-corpus training.** `train.py` already supports it and it has
never been used:

```bash
--init checkpoints/v7_63800.pt \
--repos Chanrith123333/khmer_english_ocr_image_line,SoyVitou/KhmerSynthetic1M,KimkosalYon/khmer-ocr-200k-siemreap-arial,SoyVitou/khmer-handwritten-dataset-4.2k,SoyVitou/KhmerST-Dataset-OCR-Cropped \
--weights 0.35,0.25,0.15,0.15,0.10 \
--steps 15000 --lr 1e-5
```

Oversampling handwriting to 15% of each batch is what makes it visible at all.
Keeping printed at 60% is what stops catastrophic forgetting -- the mechanism
that took `printed_B` from 1.31% to 16.13% in the sequential handwriting
fine-tune.

**Train only on the 3,362-image handwriting split** (`src/finetune_handwriting.py`,
seed 42). The other 840 are the test set. Training on them destroys the only
honest measurement in the project.

**Do not train on `62k-images-khmer-printed-dataset`** -- it is `printed_C`.

**1b. Font diversity.** `seanghay/khmerfonts-info-previews` has 2,972 Khmer
fonts and is unused. Rendering the same text corpus through hundreds of fonts
attacks the root cause directly: the model cannot learn "this exact glyph
bitmap" if the glyph changes every batch. `src/gen_chars.py` exists for this and
has never been run. **This is probably the highest-value unused asset in the
project.**

### Tier 2 — augmentation that simulates reality (cheap, no new data)

Current augmentation is noise and brightness only. Add:

| augmentation | what it fixes |
|---|---|
| random invert | light-on-dark signage, chalk, whiteboards |
| perspective warp (small homography) | photos are never flat |
| shadow / vignette overlay | uneven lighting |
| JPEG quality 30-70 | phone-camera artifacts |
| elastic distortion | handwriting stroke variation |
| random background paste | text on wood, wall, paper texture |
| stroke width (MinFilter/MaxFilter) | pen vs marker vs print |

`src/finetune_handwriting.py` already implements rotation, stroke weight, blur
and paper noise -- those are applied only in the handwriting run and should be
moved into the main training path.

### Tier 3 — capacity and resolution

9.02M parameters, of which the CNN holds 63.6%. Measured per-character error:

```
marks above/below the baseline   3.68%
base consonants                  1.24%
```

Diacritics fail 3x more often than consonants. The CNN pools height 64 -> 32 ->
16 -> 8 -> 4, so a mark occupying the top 15% of a line is under one pixel of
vertical resolution by the final layer. Options, in order of cost:

- raise `H` from 64 to 96 (more vertical detail; retrains from scratch)
- widen the CNN channels (more capacity for multi-domain)
- both, if the single-model goal is serious

### Tier 4 — decoding and post-processing (polish only)

Measured on `v7_63800`:

```
                printed_A    camera    handwriting
greedy             0.44%     80.75%        81.08%
+ normalize        0.37% ✅   81.09%        80.87%
+ TTA 3 scales     0.55% ❌   80.75%        81.08%
```

- ~~**Apply `normalize_khmer()` to every prediction.**~~ **Measured 2026-10-15:
  it is worth nothing.** `src/evaluate.py` now reports normalized CER beside raw
  CER, and over 1,840 lines v7 moves 0.5214 -> 0.5219 and v8 0.5289 -> 0.5280 --
  noise, in both directions. The reason is that every label is already passed
  through `normalize_khmer()` when the data is built (`row_sample`,
  `make_testset.py`), so the model was trained on normalized targets and emits
  normalized text. The earlier 0.44% -> 0.37% figure was measured on a
  200-line subset and did not survive the full test set.
- **Dictionary post-correction** -- snapping near-misses to a Khmer lexicon.
  Helps at 40-50% CER, useless at 80%.
- **Beam search** instead of greedy argmax -- typically 5-15% relative.
- **None of these move a broken domain.** 80% stays 80%.

---

## 4. Recommended sequence

1. **Joint multi-corpus run** from `v7_63800`, 15,000 steps, ~3h on T4 x2.
   Expect `printed_A` 0.33% -> ~2%, handwriting 87% -> ~45%, overall 52% -> ~25%.
2. **Add Tier 2 augmentation**, rerun. Expect camera to move for the first time.
3. **Font-rendered corpus** from the 2,972-font set, mixed in at 20-30%.
4. Only then consider raising `H` or widening the CNN.

Steps 1-3 are the ones that matter. Step 4 is a different project.

---

## 5. The honest expectation

Peak accuracy will get worse. `printed_A` at 0.33% is the best number in the
project and joint training will cost most of it. What you buy is a model that
reads a photo of a wall instead of one that reads a single font flawlessly.

The strongest result for the report is not a low number -- it is the
demonstration that 63,800 steps on one corpus produced 0.33% on that corpus and
87.70% on handwriting, and that fixing it required changing the data rather than
the training.
