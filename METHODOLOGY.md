# Khmer OCR — Methodology

Text recognition for Khmer script, trained from scratch (no pretrained weights).

---

## 1. Task definition

**Text recognition**, not detection. The model receives a cropped image
containing one line of Khmer text and outputs the character sequence. It does
not locate text within a full page.

This is determined by the data: all eight corpora are pre-cropped text images
with string labels. Detection would require bounding-box annotations, which
none of them provide. A full-page OCR system needs both a detector and a
recognizer; this project is the recognizer.

## 2. Data

| Source | Samples | Type | Use |
|---|---|---|---|
| SoyVitou/KhmerSynthetic1M | ~1,000,000 | synthetic | train |
| KimkosalYon/khmer-ocr-200k-siemreap-arial | ~200,000 | synthetic, single font | train |
| Sokheng/khmer-synthetic-ocr-v1-100k | ~100,000 | synthetic | train |
| SoyVitou/62k-images-khmer-printed-dataset | ~62,000 | printed | train |
| Darayut/khmer-scene-text-synthetic-contrast | — | synthetic scene | train |
| Darayut/khmer-document-synthetic-low-res | — | synthetic document | train |
| SoyVitou/khmer-handwritten-dataset-4.2k | 4,200 | **real handwriting** | **held out** |
| SoyVitou/KhmerST-Dataset-OCR-Cropped | — | **real camera** | **held out** |

**~99% of the corpus is synthetic.** This is the defining constraint of the
project and it drives three decisions: the augmentation policy (§3.3), the
choice of evaluation set (§6), and the main stated limitation (§8).

**The two real datasets are never trained on.** A model trained on rendered
fonts and evaluated on rendered fonts will report near-zero error while being
unusable on a photograph — it has learned the renderer, not the script. Real
data is the only instrument that measures the thing we actually care about, so
spending it on training would destroy the ability to measure.

## 3. Preprocessing

### 3.1 Label normalization

Khmer encodes visually identical text in multiple codepoint sequences. Without
normalization the model is asked to distinguish strings with no pixel-level
difference — an unlearnable task that suppresses accuracy across the board.
Four ambiguities are resolved (`src/normalize.py`):

1. **Subscript (COENG) ordering.** A consonant carrying two stacked subscripts
   is written `U+17D2 C1 U+17D2 C2`. Sources disagree on order; rendering is
   identical. Subscripts are sorted into a canonical order.
2. **Deprecated inherent vowels** `U+17B4`, `U+17B5`. Invisible, semantically
   empty for OCR, inconsistently present. Removed.
3. **Zero-width characters** (ZWSP, ZWNJ, ZWJ, BOM). Invisible. Removed.
4. **Intra-syllable ordering.** Register shifters, vowels and diacritics are
   emitted in varying order. Canonical order enforced:
   base → subscripts → shifter → vowel → signs.

Normalization is applied once at packing time and again to model output at
evaluation. Using the same function on both sides means the reported error rate
measures recognition, not encoding mismatch.

The function is verified idempotent: `normalize(normalize(x)) == normalize(x)`.

### 3.2 Image pipeline

- Grayscale — colour carries no information for text identity
- **Height 64 px**, aspect ratio preserved, width clamped to [8, 1024]

Height 64 rather than the 32 px standard in Latin-script OCR. Khmer stacks up
to four vertical zones — superscript vowel, base consonant, subscript COENG,
and below-diacritic. At 32 px the subscript zone falls below the resolution
needed to separate visually similar consonants, and character error rate
plateaus regardless of model capacity. This doubles compute per image and is
the single most consequential preprocessing choice.

### 3.3 Augmentation

Applied to training data only, p = 0.7: Gaussian blur, sensor noise,
brightness/contrast jitter, JPEG compression artifacts.

These specifically simulate the gap between rendered and photographed text.
Given a 99% synthetic corpus, augmentation is the primary mechanism for
generalization to real images — without it the model overfits the renderer's
pixel-exact output.

### 3.4 Storage format

Packed into LMDB. The alternative — one million loose files on mounted Google
Drive — costs 10–50 ms per file open, so an epoch spends hours in I/O with the
GPU idle. LMDB reduces this to a handful of large files. This is an
infrastructure decision, not a modelling one, but it determines whether the
project is trainable within Colab's session limits at all.

## 4. Curriculum

Samples are tiered by label length: `char` (1–2), `word` (3–12),
`sentence` (13+).

**One model, not three.** Character, word and sentence recognition share one
output alphabet and one loss — a character image is a length-1 sequence, a
sentence is a length-40 sequence. There is no architectural difference between
the tasks, so training separate models per tier would triple compute and
introduce catastrophic forgetting at each handoff. The curriculum is
implemented purely as sampling weights over a single dataset.

| Stage | Epochs | char | word | sentence |
|---|---|---|---|---|
| A | 0–2 | 60% | 35% | 5% |
| B | 3–7 | 25% | 45% | 30% |
| C | 8+ | 10% | 30% | 60% |

Early tiers never reach zero. Removing them entirely causes the model to trade
glyph-level precision for sequence-level language modelling, degrading
short-sequence accuracy — the standard catastrophic-forgetting failure.

## 5. Models

Both trained from random initialization.

### 5.1 CRNN + CTC (baseline, ~8.9M parameters)

CNN backbone → 2-layer BiLSTM → CTC loss.

Its purpose is **verification, not accuracy**. It trains in roughly 2.5 hours
and answers one question: is the data pipeline correct? Gate: validation
CER < 5%. A CRNN stuck above 15% indicates a label or charset defect, and
training a larger model on defective labels would only waste more compute.

### 5.2 AttnOCR (final, ~20.6M parameters)

CNN backbone → 4-layer transformer encoder → 4-layer autoregressive
transformer decoder, cross-entropy with label smoothing 0.1.

CTC is replaced because it assumes a **monotonic left-to-right alignment**
between input columns and output characters. Khmer violates this in two ways:
subscript consonants stack vertically beneath their base rather than advancing
horizontally, and several dependent vowels render to the *left* of the
consonant they follow in the encoding. Cross-attention has no monotonicity
constraint and can attend to any input position, which is why it is the correct
choice for this script specifically.

## 6. Evaluation

- **CER** (character error rate) — Levenshtein distance / total gold characters
- **WER** — exact-sequence mismatch rate

WER is computed over whole crops rather than space-delimited tokens, because
Khmer does not delimit words with spaces. A space-based WER would be measuring
an artifact of the dataset's segmentation choices, not linguistic words.

Both models are evaluated on the **identical held-out real dataset**, making
the comparison between them valid. Reported metrics come from real handwriting
and camera images — not from synthetic validation, which would overstate
performance substantially.

## 7. Training protocol

| | CRNN | AttnOCR |
|---|---|---|
| Optimizer | AdamW, wd 0.01 | AdamW, wd 0.01 |
| LR schedule | OneCycle, peak 3e-4 | OneCycle, peak 7e-4 |
| Batch size | 256 | 128 |
| Precision | AMP fp16 | AMP fp16 |
| Grad clipping | 5.0 | 5.0 |
| Epochs | 10 | 20 |

Checkpoints save model, optimizer, scheduler and AMP scaler state every 1000
steps to Google Drive. Free Colab terminates sessions at roughly 12 hours, so a
20–30 hour run is necessarily interrupted; resumption is a requirement, not a
convenience.

## 8. Limitations

1. **Synthetic training distribution.** ~99% of training data is rendered from
   fonts. Performance on real photographs depends on augmentation bridging that
   gap, and is measured — not assumed — via the held-out real sets.
2. **Held-out real data is small** (~4.2k handwriting plus KhmerST). Confidence
   intervals on the reported CER are correspondingly wide.
3. **Recognition only.** Full-page OCR additionally requires text detection.
4. **Single-line crops.** No multi-line or layout handling.
5. **Normalization is lossy by design.** Canonical subscript ordering discards
   the original encoding order. This is correct for recognition — the orders are
   visually identical — but the original byte sequence is not recoverable.

## 9. Anticipated questions

**Why not fine-tune TrOCR or a pretrained model?**
It would very likely score better. The project specifies from-scratch training;
a pretrained comparison is the obvious extension and would quantify what
pretraining is worth on this script.

**Why is validation CER so much worse than training loss suggests?**
Validation uses real handwriting; training uses synthetic renders. That gap
*is* the synthetic-to-real transfer gap, and exposing it is the point of the
split.

**Why 64 px height when the OCR literature standard is 32?**
Khmer stacks up to four vertical zones. See §3.2.

**Why does the curriculum keep character samples in the final stage?**
Removing them causes catastrophic forgetting of glyph-level precision. See §4.

**Why is WER not computed over words?**
Khmer has no inter-word spaces. See §6.
