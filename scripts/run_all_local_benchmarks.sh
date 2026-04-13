#!/bin/bash
set -euo pipefail

PYTHONPATH=src
PY=".venv/bin/python"

for subset in FD001 FD002 FD003 FD004; do
  $PY -m ouromaintain.train \
    --dataset cmapss \
    --cmapss-root CMAPSSData \
    --cmapss-subset "$subset" \
    --model adaptive \
    --epochs 6 \
    --batch-size 128 \
    --output-dir "artifacts/cmapss_${subset,,}_adaptive"
done

for run_name in "1st_test" "2nd_test" "4th_test/txt"; do
  safe_name=$(echo "$run_name" | tr '/' '_')
  $PY -m ouromaintain.train \
    --dataset ims \
    --ims-root IMS_extracted \
    --ims-run "$run_name" \
    --model adaptive \
    --epochs 3 \
    --batch-size 64 \
    --window-size 16 \
    --stride 4 \
    --single-asset-split stratified \
    --output-dir "artifacts/ims_${safe_name}_adaptive"
done
