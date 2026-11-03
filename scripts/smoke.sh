#!/usr/bin/env bash
# M1 on the GPU machine (docs/guide-for-me.md section 2.2). Run inside tmux.
#
#   bash scripts/smoke.sh smoke   # MATH-500 problems 1-100, 1 sample, D0, base then p50c   (~5M tokens)
#   bash scripts/smoke.sh calib   # all 500 problems; continues the smoke folders              (~27M tokens in all)
#
# Raw output: $WSPREV_RUNS/d0_math500/<model>/ (log.txt, records.jsonl, texts.jsonl.gz, meta.json).
# Summaries (for git): results/<mode>_d0/<model>/. The calibration gate is in docs/plan-for-agents.md section 3:
# if a model falls outside its interval, stop and check the evaluation pipeline before anything else.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

mode="${1:-smoke}"
case "$mode" in
  smoke) limit=100 ;;
  calib) limit=500 ;;
  *) echo "usage: bash scripts/smoke.sh smoke|calib"; exit 2 ;;
esac

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
failed=""
for model in base p50c; do
  out="$WSPREV_RUNS/d0_math500/$model"
  mkdir -p "$out"
  python -m wsprev.generate --model "$model" --config D0 --bench math500 --samples 1 --limit "$limit" \
    --out "$out" 2>&1 | tee -a "$out/log.txt"
  original=unedited
  if [ "$model" = p50c ]; then original=p50c; fi
  python scripts/summarize.py "$out" --authors "$WSPREV_LOGS/exp8_math/eval/$original" --gate "$model" \
    --out "results/${mode}_d0/$model" || failed="$failed $model"
done

echo
if [ -n "$failed" ]; then
  echo "GATE FAILED for:$failed -- do not start RQ1; send the output above to Claude."
  exit 3
fi
echo "Both models inside their calibration intervals. Commit results/${mode}_d0/ and report the tok/s above."
