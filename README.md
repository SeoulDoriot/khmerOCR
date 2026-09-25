# Khmer OCR — Three Approaches to Line-Level Text Recognition

Deep Learning final project · Kirirom Institute of Technology

Khmer script text recognition: given a cropped image of one line of Khmer text,
output the character sequence. Three deep learning approaches are trained on
identical data, with identical seeds, and scored on one frozen five-domain test
set, so the comparison measures the approaches rather than the experimental
conditions.

---

## 1. Problem

Khmer is an abugida. Consonants carry stacked subscripts (COENG, `U+17D2`),
vowel signs above and below the baseline, and diacritics on top of those. A
single orthographic cluster can be four or five codepoints tall. This makes it
harder than Latin OCR in two specific ways:

- **Vertical detail matters.** Dropping one subscript changes the word.
- **Codepoint order is ambiguous.** The same visible cluster can be encoded
  several ways, so the labels must be canonicalised before training or the
  model is asked to distinguish sequences that look identical.

`src/normalize.py` handles the second problem: NFC, strip zero-width and
deprecated inherent vowels, then reorder each cluster to
`base → sorted COENG → shifter → vowel → signs`. The function is idempotent,
which is verified rather than assumed.

## 2. Data

Ten corpora, documented in full in [DATASETS.md](DATASETS.md).

| Source | Samples | Type | Use |
|---|---|---|---|
| SoyVitou/KhmerSynthetic1M | ~1,000,000 | synthetic | train |
| Chanrith123333/khmer_english_ocr_image_line | ~12,100,000 | synthetic prose | train |
| KimkosalYon/khmer-ocr-200k-siemreap-arial | ~200,000 | synthetic, one font | train |
| Sokheng/khmer-synthetic-ocr-v1-100k | ~100,000 | synthetic receipts | train |
| SoyVitou/62k-images-khmer-printed-dataset | ~62,000 | printed | train |
| SoyVitou/khmer-handwritten-dataset-4.2k | 4,200 | **real handwriting** | **held out** |
| SoyVitou/KhmerST-Dataset-OCR-Cropped | — | **real camera** | **held out** |

**About 99% of the available data is synthetic.** That single fact drives most
of the design. A model trained on rendered fonts and evaluated on rendered
fonts reports near-zero error while being useless on a photograph — it has
learned the renderer, not the script. So the two real corpora are never trained
on by any approach; they exist to measure generalisation.

### The frozen test set

`src/make_testset.py` builds `results/test_set.pt` once and refuses to
regenerate it. 1,840 samples across five domains:

| Domain | n | Seen in training | What it is |
|---|---|---|---|
| `printed_B` | 300 | yes | synthetic printed lines |
| `printed_A` | 300 | yes | synthetic, single font |
| `printed_C` | 200 | **no** | synthetic, different generator |
| `camera` | 200 | **no** | real photographs of signage |
| `handwriting` | 840 | **no** | real handwritten lines |

Three of five domains are unseen, which is what separates generalisation from
memorisation.

## 3. Approaches

All three are implemented in PyTorch, trained by the same harness
(`src/train_approach.py`) on the same 20,000-sample subset with seed 42.

**A1 — CRNN + CTC, from scratch.** CNN (4 blocks, 64→512 channels) → 2-layer
BiLSTM → CTC. 9.02M parameters. The standard line-recognition baseline.

**A2 — CRNN + CTC, transfer.** Identical architecture, warm-started from a
checkpoint pretrained on printed Khmer, then fine-tuned at `lr 5e-5`.

**A3 — Attention encoder-decoder.** CNN → TransformerEncoder →
TransformerDecoder, autoregressive, 20.70M parameters. A structurally different
family, included so the comparison is not three variants of one idea.

### The CTC timestep constraint

CTC requires input timesteps ≥ ~2× label length. The CNN downsamples width by
4, so `W=512` gives 128 timesteps and supports labels up to 60 characters. The
first version used `W=256` — 64 timesteps against a 60-character limit — which
is mathematically unlearnable and capped CER at 45.58%. Fixing it to `W=512`
moved CER to 7.92% with no other change. The harness now asserts
`2 * max_len <= W // 4` at startup.

## 4. Results

CER on the frozen test set. Lower is better.

| Model | printed_B | printed_A | printed_C | camera | handwriting | **overall** |
|---|---|---|---|---|---|---|
| | *(seen)* | *(seen)* | *(unseen)* | *(unseen)* | *(unseen)* | |
| A1 CRNN scratch | 1.18% | 34.06% | 8.88% | 28.00% | 53.31% | **25.41%** |
| **A2 CRNN transfer** | 1.31% | **7.80%** | **5.48%** | **27.83%** | **46.79%** | **16.59%** |
| A3 Attention | 4.49% | 36.37% | 8.38% | 30.17% | 49.11% | **26.25%** |
| A2 + handwriting FT | 16.13% | 8.35% | 10.46% | 39.55% | **20.71%** | 16.92% |
| Pretrained baseline | 70.42% | **0.64%** | 51.82% | 76.97% | 86.19% | 54.43% |

| Model | Params | Train time | Best val CER |
|---|---|---|---|
| A1 | 9.02M | 62.0 min | 2.19% |
| A2 | 9.02M | 64.8 min | 2.32% |
| A3 | 20.70M | 86.4 min | 5.60% |

### Findings

**1. Validation ranking and test ranking disagree.** A1 has the better
validation CER (2.19% vs 2.32%) and loses decisively on the test set
(25.41% vs 16.59%). The entire benefit of transfer learning is invisible
in-distribution and appears on `printed_A`, 34.06% → 7.80%. Model selection on
validation alone would have picked the worse model.

**2. It is not overfitting — it is domain shift.** Validation loss falls
monotonically in all three runs and never turns up; the train/val gap stays
small (A1 ends 0.064 vs 0.096). The model generalises well to held-out data
*from the same distribution* and fails on different domains. Regularisation and
early stopping address overfitting and would not help here.

**3. Specialisation is zero-sum at this capacity.** Fine-tuning on handwriting
moves it 46.79% → 20.71%, while `printed_B` collapses 1.31% → 16.13% —
*with* 30% replay of printed data. In 9M parameters you get a printed reader or
a handwriting reader, not both.

**4. The pretrained baseline is the cautionary result.** 0.64% on `printed_A`,
better than anything else in the table, and 86.19% on handwriting. A model can
look near-perfect and be unusable one domain over. This is why the test set has
five domains.

**5. Aspect ratio predicts accuracy; image height does not.** Across all 1,840
samples, correlation(height, CER) = −0.017, correlation(aspect, CER) = −0.385.

| Aspect ratio (w/h) | n | CER |
|---|---|---|
| 0–3 | 779 | 37.92% |
| 3–6 | 446 | 34.66% |
| 6–10 | 210 | 10.16% |
| 10–16 | 99 | **3.77%** |

`prep()` scales to height 64 then right-pads to width 512 — aspect ratio 8. A
square crop becomes ~64px of content followed by 448px of blank, so most of the
128 timesteps carry nothing. This is a preprocessing defect, not a model one.

**6. Small text fails off a cliff between 20px and 16px.** Same 200 held-out
images, degraded to different pixel heights:

| Text height | CER | Exact lines |
|---|---|---|
| original | 2.32% | 117/200 |
| 36px | 3.26% | 105/200 |
| 28px | 5.37% | 80/200 |
| 20px | 16.34% | 31/200 |
| 16px | 40.22% | 1/200 |
| 12px | 74.23% | 0/200 |

## 5. Reproducing

```bash
pip install -r requirements.txt

python -u src/make_testset.py              # frozen test set (writes once)
python -u src/make_trainset.py --n 20000   # frozen train subset

python -u src/train_approach.py --arch crnn --tag A1_crnn_scratch
python -u src/train_approach.py --arch crnn --tag A2_crnn_transfer --init <ckpt>
python -u src/train_approach.py --arch attn --tag A3_attn_scratch

python -u src/evaluate.py                  # one protocol, all models
python -u src/figures.py                   # learning curves + comparison
```

Every run calls `set_all_seeds(42)` (`src/seeds.py`): `random`, `numpy`,
`torch`, `torch.cuda`, and `PYTHONHASHSEED`.

### Inspection tools

```bash
python -u src/dataset_preview.py                     # images + transcripts
python -u src/predict_gallery.py --ckpt <ckpt>       # gold vs prediction, per domain
python -u src/test_live.py <ckpt> handwriting 8      # one image at a time
python -u src/make_small_text.py --mode recognizer   # small-text augmentation
```

## 6. Limitations

- **~99% synthetic training data.** Every number on a real-image domain should
  be read as the cost of that choice.
- **Recognition only.** The model reads a cropped line; it does not find text
  in a page. `src/page_ocr.py` segments by projection profile, which works on
  clean scans and fails on photographs.
- **Camera domain is unsolved** at ~28% CER for every approach.
- **4,000-step budget per approach.** Comparable across approaches, but none is
  trained to convergence. The separately trained 39,400-step model reaches
  0.68% CER in-distribution — not comparable to the cross-domain numbers above.

## 7. Use of AI assistance

Claude (Anthropic) was used throughout as a coding and analysis assistant:
writing and debugging training and evaluation scripts, diagnosing the CTC
timestep bug, designing the frozen-test-set protocol, and running the
measurements in §4. All experiments were executed by the author; all reported
numbers come from the committed scripts and can be regenerated with the
commands above. Claude's analysis was corrected by measurement at least once —
an initial claim that image *height* drove the camera failures was refuted by
the full-test-set correlation in finding 5.

## 8. Citations

Datasets are credited in [DATASETS.md](DATASETS.md) with sources and licences.

- Shi, Bai & Yao (2015), *An End-to-End Trainable Neural Network for
  Image-Based Sequence Recognition* — CRNN.
- Graves et al. (2006), *Connectionist Temporal Classification* — CTC.
- Vaswani et al. (2017), *Attention Is All You Need* — A3's decoder.
