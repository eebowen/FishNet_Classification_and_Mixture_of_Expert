# Multi-Head Baseline (no Experts)

A simple baseline to compare with MMoE: each task has its own 2-layer head applied directly on cached DINOv3 features.

- Heads: LayerNorm -> Linear -> GELU -> Dropout -> Linear
- Tasks: Family (multiclass), Order (multiclass), Habitat (8-label multilabel), Troph (regression)
- Optional: FeedingPath (categorical) with --include-feedingpath

## Train
```bash
python -m dinov3.projects.multi_head.train_multihead \
  --features-dir artifacts/fishnet2_balanced/features \
  --ann-root datasets/anns \
  --output-dir artifacts/multihead_runs \
  --epochs 100 --batch-size 256 --lr 1e-3 \
  --lr-scheduler linear --lr-warmup-epochs 5 \
  --tower-hidden 1024 --dropout 0.1 \
  --class-weighting balanced --standardize --wandb
```

Metrics match the MMoE script for apples-to-apples comparison (accuracy, macro F1@0.5, MAE, composite score).
