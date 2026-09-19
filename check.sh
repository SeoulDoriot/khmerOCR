#!/bin/bash
cd /Users/seouldorito/khmer-ocr
echo "=== progress ==="
grep -E "^step|>> CER|stopped at|exited rc|attempt|charset=" logs/overnight.log | tail -16
echo
echo "=== status ==="
pgrep -f train_mac.py >/dev/null && echo "TRAINING (pid $(pgrep -f train_mac.py))" || echo "NOT running"
pgrep -f run_overnight.sh >/dev/null && echo "wrapper alive (will auto-restart)" || echo "wrapper stopped"
echo
echo "=== checkpoints ==="
ls -lh checkpoints/ 2>/dev/null | tail -4
echo
echo "=== best so far ==="
grep "<- best" logs/overnight.log | tail -3
