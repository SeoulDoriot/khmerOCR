#!/bin/bash
# Live status of every local training run.  Usage:  ./status.sh   or   ./status.sh -f
cd /Users/seouldorito/khmer-ocr

if [ "$1" = "-f" ]; then
  L=$(ls -t logs/*.log | head -1)
  echo "following $L  (ctrl-C to stop)"; echo
  exec tail -f "$L"
fi

for f in logs/handwriting.log logs/overnight.log logs/approach_*.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .log)
  echo "=== $name ==="
  grep -E ">> CER|CER .* ->" "$f" | tail -3
  grep -E "^step" "$f" | tail -1
  echo
done

echo "=== processes ==="
for p in finetune_handwriting train_local train_approach train_mac; do
  pgrep -f "$p.py" >/dev/null && echo "  RUNNING: $p" 
done
pgrep -f "\.py" >/dev/null || echo "  nothing training locally"

echo
echo "=== checkpoints ==="
ls -lh checkpoints/*.pt 2>/dev/null | awk '{print "  "$5"  "$9"  "$6" "$7" "$8}'
