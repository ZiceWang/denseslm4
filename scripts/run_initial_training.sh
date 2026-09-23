#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
exec python -m denseslm4.train_moe \
  --stride 896 \
  --warmup-steps 9000 \
  --scheduler wcl \
  --output-dir runs/denseslm4_moe_v2 \
  2>&1 | tee runs/denseslm4_moe_v2/training.log
