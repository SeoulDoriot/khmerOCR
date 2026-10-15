# Printed and handwriting datasets

## Choose the correct image category

- **Printed Khmer:** 10000 selected exact-match images in the printed/ folder of [this archive](https://huggingface.co/datasets/Seoul12/khmer-ocr-selected-demo/resolve/main/Khmer_OCR_10000_Selected.zip?download=true). Uses the saved Transformer at step 87000. Open printed/Gallery_10000.html after unzipping.
- **Khmer handwriting:** 151 visually reviewed handwriting examples in the handwriting/ folder of [the separate handwriting archive](https://huggingface.co/datasets/Seoul12/khmer-ocr-handwriting-demo/resolve/main/Handwriting_Khmer_Selected.zip?download=true). Uses the available handwriting BiLSTM fine tune at step 3000. Every selected handwriting example was rerun individually on CPU with an exact match. Open handwriting/Gallery_Handwriting.html after unzipping.

[Open the category test notebook in Colab](https://colab.research.google.com/github/SeoulDoriot/khmerOCR/blob/main/Test_By_Category.ipynb). Run all, select Printed or Handwriting, click Load selected dataset, enter the example number from that gallery and click Run real prediction. Changing category requires loading the matching dataset again; the notebook prevents testing with the previously loaded wrong model.

These are selected demonstration sets, not overall model accuracy on new images. The handwriting source was mixed: 840 project-partition candidates included printed and English examples too. We inspected all 222 Khmer exact-match candidates and retained 151 clear handwriting examples. Obvious printed/English examples and uncertain tiny glyphs were excluded. Writer provenance is not established, and the partition was also used for fine-tune checkpoint selection. The handwriting checkpoint is the local BiLSTM variant, not the unavailable Kaggle Transformer specialist. Do not use the mixed-pool CER as a handwriting-only benchmark.

Printed source: Chanrith123333/khmer_english_ocr_image_line, CC BY 4.0. Handwriting source: SoyVitou/khmer-handwritten-dataset-4.2k, MIT declared in upstream metadata; upstream card is included. Source attribution, checksums, answers, predictions and review records are preserved in each archive.

