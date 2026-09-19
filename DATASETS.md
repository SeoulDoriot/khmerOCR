# Datasets

All data is public. Every dataset used in this project is listed below with its
source, licence and the role it plays. Licence and size figures were read from
the Hugging Face dataset API on 2026-09-19.

## Training data

| Dataset | Size | Licence | Link | Role |
|---|---|---|---|---|
| `Chanrith123333/khmer_english_ocr_image_line` | 12,138,214 | CC-BY-4.0 | [HF](https://huggingface.co/datasets/Chanrith123333/khmer_english_ocr_image_line) | Primary corpus. Natural Khmer prose plus some English lines. |
| `SoyVitou/KhmerSynthetic1M` | ~1M | Apache-2.0 | [HF](https://huggingface.co/datasets/SoyVitou/KhmerSynthetic1M) | Second corpus from a different generator. Used to test and fix over-specialisation. |
| `KimkosalYon/khmer-ocr-200k-siemreap-arial` | 200,000 | MIT | [HF](https://huggingface.co/datasets/KimkosalYon/khmer-ocr-200k-siemreap-arial) | Single-font (Siemreap Arial) literary and legal Khmer. |
| `SoyVitou/62k-images-khmer-printed-dataset` | 62,000 | MIT | [HF](https://huggingface.co/datasets/SoyVitou/62k-images-khmer-printed-dataset) | Book-style printed text with punctuation. |
| `Sokheng/khmer-synthetic-ocr-v1-100k` | 100,000 | CC-BY-4.0 | [HF](https://huggingface.co/datasets/Sokheng/khmer-synthetic-ocr-v1-100k) | Synthetic receipts, menus, price tags. Used for the from-scratch baseline. |
| `Darayut/khmer-scene-text-synthetic-contrast` | 100K–1M | not stated | [HF](https://huggingface.co/datasets/Darayut/khmer-scene-text-synthetic-contrast) | Scene-text conditions. Prepared, not used in final runs. |
| `Darayut/khmer-document-synthetic-low-res` | 100K–1M | not stated | [HF](https://huggingface.co/datasets/Darayut/khmer-document-synthetic-low-res) | Low-resolution documents. Prepared, not used in final runs. |
| `SoyVitou/khmer-handwritten-dataset-4.2k` | 4,202 | MIT | [HF](https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k) | **Real handwriting.** 80% (3,362) used for stage-2 fine-tuning. |

## Held-out test data — never trained on

| Dataset | Size | Licence | Link | Role |
|---|---|---|---|---|
| `SoyVitou/khmer-handwritten-dataset-4.2k` (20% split) | 840 | MIT | [HF](https://huggingface.co/datasets/SoyVitou/khmer-handwritten-dataset-4.2k) | Handwriting test split, seed 42. |
| `SoyVitou/KhmerST-Dataset-OCR-Cropped` | 1K–10K | Apache-2.0 | [HF](https://huggingface.co/datasets/SoyVitou/KhmerST-Dataset-OCR-Cropped) | Real camera photographs of Khmer text. |

## Also referenced

| Dataset | Link | Note |
|---|---|---|
| `mrrtmob/khmer_english_ocr_image_line` | [HF](https://huggingface.co/datasets/mrrtmob/khmer_english_ocr_image_line) | Original of the 12.1M corpus. **Gated** (requires accepting terms), so the ungated `Chanrith123333` mirror was used instead. |
| `emhengly/khmer-text-recognition-dataset` | [Kaggle](https://www.kaggle.com/datasets/emhengly/khmer-text-recognition-dataset) | Downloaded during exploration, not used. |
| `seanghay/khmerfonts-info-previews` | [HF](https://huggingface.co/datasets/seanghay/khmerfonts-info-previews) | 2,972 Khmer fonts. Reference for font variety, not trained on. |

## The frozen test set

`results/test_set.pt` (1,840 samples) is built once by `src/make_testset.py` with
seed 42 and is used to evaluate **every** approach, so the comparison is fair.

| Domain | n | Source | Seen in training |
|---|---|---|---|
| `printed_A` | 300 | 12.1M corpus | yes |
| `printed_B` | 300 | KhmerSynthetic1M | yes |
| `printed_C` | 200 | 62k printed | no |
| `handwriting` | 840 | handwritten-4.2k, 20% split | no |
| `camera` | 200 | KhmerST cropped | no |

Three of the five domains were never trained on by any approach, which lets the
comparison measure generalisation rather than memorisation.

## Known limitations of the data

- **~99% of training data is synthetic** (rendered from fonts). Real handwriting
  and camera photographs are a different visual distribution, and the reported
  gap between printed and handwriting accuracy is largely a consequence of this.
- **Only 4,202 real handwriting samples exist** across every source found. This
  is roughly 4,000x less data than the printed corpora and is the main limit on
  handwriting accuracy.
- **Two datasets state no licence** (`Darayut/*`). They were not used in the
  final experiments.
- **Label noise exists.** The 12.1M corpus contains generation artefacts such as
  `# Entry 62654 - TD1_ID_CARD` leaking into label text; the Khmer-majority
  filter removes most but not all.
