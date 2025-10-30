#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=3 python -m dinov3.projects.multitask_moe.train_mmoe \
  --features-dir artifacts/fishnet2_balanced/features \
  --ann-root datasets/anns \
  --output-dir artifacts/mmoe_runs \
  --epochs 100 \
  --batch-size 256 \
  --lr 2e-3 \
  --weight-decay 0.0 \
  --lr-scheduler linear \
  --lr-warmup-epochs 5 \
  --experts 8 \
  --expert-hidden 2048 \
  --tower-hidden 1024 \
  --dropout 0.1 \
  --loss-weights 1.0 0.5 0.5 0.5 \
  --class-weighting balanced \
  --standardize \
  --wandb \
  --wandb-project fishnet_moe \
  --patience 50