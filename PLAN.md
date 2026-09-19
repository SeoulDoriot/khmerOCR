# Khmer OCR — From-Scratch Training Plan

Target: one model, character -> word -> sentence curriculum, ~1M samples,
trained from scratch on Colab. Beat a small baseline, end with a large model.

---

## Phase 0 — Data audit (CPU only, ~30 min, do this FIRST)

No GPU. This phase exists because most from-scratch OCR failures are label
bugs, and finding one after a 10-hour run is the expensive way to learn it.

Produce:
- `charset.json` — sorted list of every Unicode codepoint in the labels
- `stats.json` — count per tier (char/word/sentence), label length histogram,
  image width/height histogram
- `suspects.csv` — labels containing codepoints seen fewer than 20 times

Gate: charset size should land ~80-140 codepoints. If it is 300+, labels are
not normalized and Phase 1 normalization is wrong — fix before continuing.

---

## Phase 1 — Preprocessing

### Label normalization (Khmer-specific, non-negotiable)

Khmer encodes identical-looking text multiple ways. Without normalization the
model is asked to learn distinctions that do not exist in the pixels.

1. Unicode NFC
2. Canonical COENG ordering — for a base consonant with multiple U+17D2
   subscripts, sort the subscript sequence deterministically
3. Strip zero-width characters: U+200B (ZWSP), U+200C (ZWNJ), U+FEFF
4. Normalize the vowel-reordering cases (U+17B6 vs decomposed forms)
5. Map any Latin/digit characters to a single consistent casing policy

Every label passes through the *same* function at train and inference time.
Ship it as one module, import it in both places, never inline it.

### Image pipeline

- Grayscale
- **Height 64**, not the Latin-OCR default of 32. Khmer stacks up to four
  vertical zones (superscript vowel / base / subscript COENG / below-diacritic).
  At 32px the subscripts turn to mush and CER plateaus around 8-10%.
- Width: preserve aspect ratio, clamp to [32, 512], right-pad to bucket width
- Normalize to mean 0.5 / std 0.5

### Augmentation (this is what makes synthetic data transfer)

If the 1M samples are rendered from fonts, the model will hit ~0% CER on
val and fall apart on a real photo. Augmentation is the fix, applied at
p=0.7 per sample:

perspective warp, gaussian blur, gaussian noise, JPEG compression artifacts,
brightness/contrast jitter, slight elastic distortion, background texture blend

### Packing

1M loose files on mounted Drive is the single biggest performance mistake
available here — per-file Drive latency is 10-50ms, so an epoch spends hours
waiting on I/O with the GPU idle.

Pack into **LMDB shards, ~1GB each**, copy shards to local `/content` disk at
the start of each session. Expect 10-50x throughput improvement.

### Splits

- train 94% / val 3% / test 3%
- Split by **source text**, not randomly. If the same sentence is rendered in
  12 fonts, all 12 renders must land in the same split or val is leaking.

---

## Phase 2 — Baseline: CRNN + CTC

Purpose is not accuracy. Purpose is proving the pipeline is correct, cheaply.

- CNN backbone (ResNet-ish, ~7M) -> BiLSTM 2x256 -> CTC over charset
- ~15M params
- AdamW, lr 3e-4, cosine decay, weight decay 1e-4
- Batch 256, AMP (fp16)
- 10 epochs, ~15 min/epoch on a T4 => ~2.5 hours total

Gate: val CER < 5%. If CER is stuck above 15%, stop — it is a label or
charset bug, not a capacity problem. Go back to Phase 0.

---

## Phase 3 — The big model: PARSeq

Attention decoder, not CTC. CTC assumes monotonic left-to-right alignment;
Khmer subscripts and reordered vowels violate that, and it costs you real
accuracy on exactly the hard cases.

- PARSeq, ViT-small encoder, ~24M params (scale to ~100M if the T4 allows and
  Phase 2 showed the data supports it)
- AdamW, lr 7e-4, OneCycle schedule, weight decay 0.01
- Label smoothing 0.1
- Batch 128, AMP
- 20 epochs, ~1-1.5 hr/epoch on a T4 => 20-30 hours

### Curriculum (one model, staged data mix)

| Stage  | Epochs | char | word | sentence |
|--------|--------|------|------|----------|
| A      | 1-3    | 60%  | 35%  | 5%       |
| B      | 4-8    | 25%  | 45%  | 30%      |
| C      | 9-20   | 10%  | 30%  | 60%      |

Earlier tiers never drop to zero. That is what prevents forgetting.
Implement as sampler weights over one dataset, not three dataloaders.

---

## Phase 4 — Evaluation

- CER and WER on test, reported **per tier** (char / word / sentence)
- Head-to-head vs the Phase 2 CRNN on the identical test set
- Confusion analysis restricted to COENG clusters — that is where the
  remaining errors will concentrate, and it tells you whether to add data
  or add capacity

---

## Colab survival rules

Free Colab disconnects at ~12 hours and reclaims GPUs without warning.
Phase 3 is a 20-30 hour run, so it *will* be interrupted. Non-negotiable:

- Checkpoint every 1000 steps to Drive (not local disk)
- Save optimizer + scheduler + sampler state, not just weights
- Training script resumes from latest checkpoint automatically on start
- Keep only the last 3 checkpoints plus the best

A run that cannot resume is a run that never finishes.

---

## Compute budget

| Phase | Hardware | Time |
|-------|----------|------|
| 0 audit | CPU | 0.5 h |
| 1 packing | CPU | 2-4 h (one time) |
| 2 CRNN | T4 | 2.5 h |
| 3 PARSeq | T4 | 20-30 h (across sessions) |
| 4 eval | T4 | 1 h |

Colab Pro (L4/A100) cuts Phase 3 to roughly 6-10 hours. That is the only
place paying actually changes the plan.
