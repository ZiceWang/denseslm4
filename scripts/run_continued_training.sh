#!/bin/bash
cd /data1/neu_lab2/denseslm4
source .venv/bin/activate
exec python -m denseslm4.train_moe \
  --num-train-epochs 0.15 \
  --learning-rate 1e-5 \
  --muon-lr 5e-4 \
  --warmup-steps 1000 \
  --resume-from-checkpoint runs/denseslm4_moe/final_model \
  --output-dir runs/denseslm4_moe_continued \
  --logging-steps 100 \
  2>&1 | tee runs/denseslm4_moe_continued/training.log