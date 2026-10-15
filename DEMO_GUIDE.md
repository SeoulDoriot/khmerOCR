# Teacher demonstration guide

## Choose the correct image category

- **Printed Khmer:** 10000 selected exact-match images in the printed/ folder of [this archive](https://huggingface.co/datasets/Seoul12/khmer-ocr-selected-demo/resolve/main/Khmer_OCR_10000_Selected.zip?download=true). Uses the saved Transformer at step 87000. Open printed/Gallery_10000.html after unzipping.
- **Khmer handwriting:** 151 visually reviewed handwriting examples in the handwriting/ folder of [the separate handwriting archive](https://huggingface.co/datasets/Seoul12/khmer-ocr-handwriting-demo/resolve/main/Handwriting_Khmer_Selected.zip?download=true). Uses the available handwriting BiLSTM fine tune at step 3000. Every selected handwriting example was rerun individually on CPU with an exact match. Open handwriting/Gallery_Handwriting.html after unzipping.

[Open the category test notebook in Colab](https://colab.research.google.com/github/SeoulDoriot/khmerOCR/blob/main/Test_By_Category.ipynb). Run all, select Printed or Handwriting, click Load selected dataset, enter the example number from that gallery and click Run real prediction. Changing category requires loading the matching dataset again; the notebook prevents testing with the previously loaded wrong model.

These are selected demonstration sets, not overall model accuracy on new images. The handwriting source was mixed: 840 project-partition candidates included printed and English examples too. We inspected all 222 Khmer exact-match candidates and retained 151 clear handwriting examples. Obvious printed/English examples and uncertain tiny glyphs were excluded. Writer provenance is not established, and the partition was also used for fine-tune checkpoint selection. The handwriting checkpoint is the local BiLSTM variant, not the unavailable Kaggle Transformer specialist. Do not use the mixed-pool CER as a handwriting-only benchmark.

Printed source: Chanrith123333/khmer_english_ocr_image_line, CC BY 4.0. Handwriting source: SoyVitou/khmer-handwritten-dataset-4.2k, MIT declared in upstream metadata; upstream card is included. Source attribution, checksums, answers, predictions and review records are preserved in each archive.

## 10000 selected Transformer demonstration images

[Open the 10000 image live demo in Colab](https://colab.research.google.com/github/SeoulDoriot/khmerOCR/blob/main/Demo_10000.ipynb). Run all, enter an example number, and click Run real prediction. CPU is sufficient.

[Download the 10000 image gallery](https://huggingface.co/datasets/Seoul12/khmer-ocr-selected-demo/resolve/main/Khmer_OCR_10000_Selected.zip?download=true). Unzip it and open printed/Gallery_10000.html. Browse, search Khmer answers, or jump to an example number. Use that same number in the notebook. Each selected image has its original normalized answer, source validation row and image checksum.

These are **10000 selected exact-match printed line examples for the Transformer only**. This is not 100% accuracy on arbitrary images or a BiLSTM/BiGRU selection. The model weights are unchanged. First 500 candidates were screened on CPU and later candidates on Apple MPS GPU; 32-image initial token parity and 100 random single-image CPU rechecks passed. Numerical differences across runtimes remain possible. All source answers were preserved with NFC/whitespace normalization; no answer was replaced with a prediction. Full screening results, including errors, are in screening_results.jsonl inside the ZIP.

## Expanded model specific gallery

Screened **500 unique printed validation images** using the three saved models. Selected exact matches: **Transformer 380**, **BiLSTM 273**, **BiGRU 70**. **49 images were correct for all three.** These sets overlap; there are 426 selected unique images in total. These are selected demonstration examples, not overall accuracy or a fair benchmark.

[Open expanded live demo in Colab](https://colab.research.google.com/github/SeoulDoriot/khmerOCR/blob/main/Expanded_Demo.ipynb). Run all, choose a model and image, then click Run real prediction. CPU is sufficient.

[Download the gallery and image folders](https://huggingface.co/datasets/Seoul12/khmer-ocr-selected-demo/resolve/main/Expanded_Khmer_OCR_Demo.zip?download=true). Unzip it and open Gallery.html. Choose a model, or choose Correct for all three models. The all_three folder contains the 49 common exact-match images; individual model folders include more examples and answers.csv.

Images are cropped printed text lines. New images, handwriting and full pages can behave differently. Labels were kept from the source dataset and normalized consistently, not replaced with predictions. expanded_screening_results.json includes failures. The earlier 20-image demo below remains available.

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
