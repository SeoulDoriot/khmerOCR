#!/bin/bash
# Train all three approaches sequentially on the shared subset.
cd /Users/seouldorito/khmer-ocr
source /Users/seouldorito/cambodian-name-tts/.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1

# A3 first: longest and least certain, so failures surface early
python -u src/train_approach.py --arch attn --tag A3_attn_scratch \
    --steps 4000 --batch-size 32 > logs/approach_A3.log 2>&1

python -u src/train_approach.py --arch crnn --tag A1_crnn_scratch \
    --steps 4000 --batch-size 32 > logs/approach_A1.log 2>&1

python -u src/train_approach.py --arch crnn --tag A2_crnn_transfer \
    --init "$HOME/Downloads/quickstart2_crnn.pt" \
    --steps 4000 --batch-size 32 > logs/approach_A2.log 2>&1

echo "ALL THREE APPROACHES DONE $(date)" >> logs/approach_A3.log
