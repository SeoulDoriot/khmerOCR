# Teacher demonstration guide

## Quick demonstration

[Open the Transformer demo in Colab](https://colab.research.google.com/github/SeoulDoriot/khmerOCR/blob/main/Khmer_OCR_Demo.ipynb). Choose Runtime → Run all. It runs on CPU and shows each image, expected answer, actual prediction and RIGHT/WRONG. A Google account may be required to execute Colab; no Hugging Face account is needed to download public files.

[Open all three architectures in Colab](https://colab.research.google.com/github/SeoulDoriot/khmerOCR/blob/main/Compare_Models.ipynb). BiLSTM, BiGRU and Transformer checkpoints are at [Hugging Face](https://huggingface.co/Seoul12/khmer-ocr-three-architectures). Optional own-image upload is disabled by default; set TRY_YOUR_IMAGE=True to enable it. Supply a cropped text line, not a full page.

Read Khmer_OCR_Project_Report.docx for architecture, training data, results and limitations.

## Results and selection

All three models passed strict weight loading and CPU inference. Saved steps: BiLSTM 58,000; BiGRU 8,500; Transformer 87,000. Weights were exported without optimizer and RNG state, so these exports do not support exact training resumption.

Twenty examples were selected after the Transformer predicted them correctly. On this selected demo set: BiLSTM 14/20, BiGRU 6/20, Transformer 20/20. This is a deliberately selected demonstration, not an independent benchmark or fair ranking. Forty original printed validation examples were screened: Transformer 28/40 exact matches. screening_results.json includes failed predictions. verification.json contains the three-model predictions.

Shared NFC and rounded-width preprocessing can differ from original recurrent training. Do not compare stored validation CER values across these runs. An independent common holdout, each model's validated preprocessing, CER and exact-line accuracy are needed for a fair benchmark.

Demo image source: Chanrith123333/khmer_english_ocr_image_line, validation split, CC BY 4.0. Manifest records source rows, normalized labels and converted grayscale PNGs. [Published dataset](https://huggingface.co/datasets/Seoul12/khmer-ocr-selected-demo).

## Separate experiments

The existing repository README reports older A1/A2/A3 approaches: BiLSTM from scratch, BiLSTM transfer and attention encoder-decoder. A1/A2 share an architecture. The newer collection has BiLSTM, BiGRU and Transformer Encoder with CTC. The older README results must not be assigned to these checkpoints.

## Training references

- [BiLSTM architecture and pilot](https://www.kaggle.com/code/seoulvy/khmer-ocr-handwriting-and-model-comparison)
- [Transformer continuation](https://www.kaggle.com/code/seoulvy/khmer-ocr-architecture-3-100k-continuation), saved version 356010649
- [Handwriting specialist](https://www.kaggle.com/code/seoulvy/khmer-ocr-handwriting-specialist), saved version 356670390

The handwriting specialist is a fine-tuned Transformer, not a fourth architecture. Its reported test results use a different evaluation set.

## Verification scope and assistance

The models and inference code were executed locally on CPU. Colab notebooks are supplied for execution; a complete run in the Colab UI has not been verified. Codex assisted with checkpoint recovery, inference code, verification, packaging and this documentation. Predictions in the verification files were computed by the models.
