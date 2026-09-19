#!/bin/bash
cd /Users/seouldorito/khmer-ocr
source /Users/seouldorito/cambodian-name-tts/.venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
LOG=logs/overnight.log
echo "=== LOCAL PARQUET run started $(date) ===" >> "$LOG"
for i in $(seq 1 60); do
  echo "--- attempt $i $(date) ---" >> "$LOG"
  python -u src/train_local.py --repo "/Users/seouldorito/.cache/huggingface/hub/datasets--Sokheng--khmer-synthetic-ocr-v1-100k/snapshots/dca80198ee5e16dea60db11bbe7b7811c973a048/data/train-*.parquet" \
      --steps 20000 --khmer-only --batch-size 32 --val-size 400 --workers 2 \
      --save-every 250 --eval-every 1000 >> "$LOG" 2>&1
  rc=$?
  echo "--- exited rc=$rc $(date) ---" >> "$LOG"
  [ $rc -eq 0 ] && break
  sleep 30
done
echo "=== finished $(date) ===" >> "$LOG"
