# Khmer OCR — Slide Outline

14 slides · ~12 min · Deep Learning final project, KIT

Every number here comes from `results/evaluation.json`, `results/curve_*.csv`,
or a script in `src/`. Figures: `results/fig1_learning_curves.png`,
`results/fig2_comparison.png`, and the HTML galleries in `results/`.

---

### 1 — Title
Khmer OCR: three deep learning approaches to line-level text recognition.
Name, course, date.

---

### 2 — The problem
Given a cropped image of one line of Khmer, output the character sequence.

Khmer is an abugida: stacked subscripts (COENG `U+17D2`), vowel signs above and
below the baseline. One cluster can be 4–5 codepoints tall.

> **Say:** two things make it harder than Latin OCR — vertical detail carries
> meaning, and the same visible cluster can be encoded several ways.

---

### 3 — Normalisation
`src/normalize.py`: NFC → strip zero-width and deprecated inherent vowels →
reorder each cluster to `base → sorted COENG → shifter → vowel → signs`.
Verified idempotent.

> **Say:** without this the model is asked to distinguish sequences that look
> identical on screen. This is a Khmer-specific step Latin OCR never needs.

---

### 4 — Data, and the honest problem with it
Ten corpora. **~99% synthetic.**

| Real data | n | Use |
|---|---|---|
| khmer-handwritten-4.2k | 4,200 | **held out** |
| KhmerST (camera) | — | **held out** |

> **Say:** a model trained on rendered fonts and tested on rendered fonts
> reports near-zero error and is useless on a photo. It learned the renderer.
> Real data is the only instrument that measures what we care about, so we
> refuse to spend it on training.

---

### 5 — The frozen test set
1,840 samples, five domains, built once by `src/make_testset.py`, which
refuses to regenerate. Three of five domains are **unseen by every approach**.

> **Say:** this separates generalisation from memorisation, and it's the same
> data for all three approaches — the comparison measures the method, not the
> conditions.

---

### 6 — Three approaches
| | Architecture | Params |
|---|---|---|
| A1 | CRNN + CTC, from scratch | 9.02M |
| A2 | CRNN + CTC, transfer | 9.02M |
| A3 | Attention encoder-decoder | 20.70M |

Same harness, same 20,000 samples, seed 42.

> **Say:** A3 is a structurally different family on purpose — otherwise this is
> three variants of one idea.

---

### 7 — The bug worth talking about
CTC needs timesteps ≥ ~2× label length.
`W=256` → 64 timesteps, `max_len=60`. **Mathematically unlearnable.**

**45.58% → 7.92% CER** from changing one constant.

The harness now asserts `2 * max_len <= W // 4` at startup.

> **Say:** no amount of training fixes this. It looks like a model problem and
> it's an arithmetic problem. This is the single biggest accuracy win in the
> project.

---

### 8 — Results
`results/fig2_comparison.png`

| Model | overall test CER |
|---|---|
| A1 CRNN scratch | 25.41% |
| **A2 CRNN transfer** | **16.59%** |
| A3 Attention | 26.25% |

---

### 9 — Finding 1: validation lies
A1 wins on validation (**2.19%** vs 2.32%).
A2 wins on test (**16.59%** vs 25.41%).

The gap is almost entirely `printed_A`: **34.06% → 7.80%**.

> **Say:** transfer learning's benefit is invisible in-distribution. Selecting
> on validation alone would have picked the worse model.

---

### 10 — Finding 2: not overfitting, domain shift
`results/fig1_learning_curves.png`

Validation loss falls monotonically in all three runs. Never turns up.

| | Overfitting | Domain shift |
|---|---|---|
| Held-out, same distribution | bad | **good (2.32%)** |
| Different domain | bad | **bad (46.79%)** |

> **Say:** we checked. Regularisation and early stopping fix overfitting and
> would not help here. The fix is data from the target domain.

---

### 11 — Finding 3: specialisation is zero-sum
Fine-tuning on handwriting, **with 30% replay**:

| | before | after |
|---|---|---|
| handwriting | 46.79% | **20.71%** |
| printed_B | 1.31% | **16.13%** |

> **Say:** in 9M parameters you get a printed reader or a handwriting reader,
> not both. This is the answer to "why not one model for everything."

---

### 12 — Finding 4: the baseline that looks perfect
Pretrained specialist: **0.64%** on printed_A — best number in the whole study.
Same model, handwriting: **86.19%**.

> **Say:** this is why the test set has five domains instead of one.

---

### 13 — Finding 5: what actually breaks it
correlation(height, CER) = **−0.017** · correlation(aspect ratio, CER) = **−0.385**

| aspect (w/h) | CER |
|---|---|
| 0–3 | 37.92% |
| 10–16 | **3.77%** |

`prep()` pads a square crop with 448px of blank. Most timesteps carry nothing.

Small text falls off a cliff: 20px → 16.34%, **16px → 40.22%**.

> **Say:** we first blamed image height. The full-set correlation refuted it.
> Both of these are preprocessing, fixable without retraining.


---

### 13b — Finding 6: scale makes it NARROWER, not better

`v7_63800` — 63,800 steps, 16x more training than A1/A2/A3.

| domain | CER | exact lines |
|---|---|---|
| printed_A *(its corpus)* | **0.33%** | **187/200** |
| printed_C | 55.69% | 3/200 |
| printed_B | 76.86% | 1/200 |
| camera | 86.66% | 2/200 |
| handwriting | 87.70% | 5/200 |

From 39,400 → 63,800 steps: `printed_A` 0.40% → 0.33%. **Every other domain
flat.**

> **Say:** 187 perfect lines on one domain, one on another. The model learned a
> *renderer*, not a *script*. Our own methodology predicted this before we
> measured it. More steps lowered the number that was already good and moved
> nothing else — the corpus is the ceiling, not the compute.

---

### 13c — Finding 7: small-text training works

Canvas expansion — grow the image, keep the text, so the text arrives smaller.

| | before | after |
|---|---|---|
| small text (expanded) | 62.21% | **26.24%** |
| normal text | 8.24% | **6.90%** |

8 of 8 sample images improved. The 20,000-step Kaggle version reached
**0.59% CER**, better than the 0.65% base it started from.

> **Say:** the augmentation acted as regularisation — it got better at normal
> text while learning small text. This is the one experiment where we changed
> the *data* rather than the *amount*, and it is the only one that moved a
> number that mattered.

---

### 13d — What would fix it (and what we measured)

Decoding tricks, measured on v7_63800:

| | printed_A | camera | handwriting |
|---|---|---|---|
| greedy | 0.44% | 80.75% | 81.08% |
| + Khmer normalisation | **0.37%** | 81.09% | 80.87% |
| + test-time augmentation | 0.55% | 80.75% | 81.08% |

> **Say:** normalisation is free and removes ~16% of remaining errors — but
> nothing in decoding moves a broken domain. 80% stays 80%. The fix is joint
> multi-corpus training: `train.py` supports per-corpus weights and every run so
> far used `--weights 1.0` on one corpus. That is the next experiment, and it is
> in IMPROVEMENT_PLAN.md.

---

### 14 — Limitations and next steps
- ~99% synthetic training data — the cost shows on every real-image domain
- Recognition only; no detector. Camera unsolved at ~28%
- 4,000 steps per approach — comparable, not converged

**Next:** symmetric padding (free), small-text fine-tuning at `lr 1e-5` with
replay, then detector → crop → recognise for full pages.

> **Close:** the headline is that the ranking changed between validation and
> test, and the only reason we saw it is that the test set had domains the
> training data didn't.
