#!/bin/bash
source /Users/seouldorito/cambodian-name-tts/.venv/bin/activate
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE
for i in $(seq 1 50); do
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Sokheng/khmer-synthetic-ocr-v1-100k", repo_type="dataset",
                      allow_patterns=["data/*.parquet"], max_workers=2)
print("COMPLETE:", p)
PY
  [ $? -eq 0 ] && break
  sleep 30
done
