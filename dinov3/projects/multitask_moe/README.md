# Multi-task Mixture-of-Experts (MMoE) for FishNet

This project trains a Mixture-of-Experts model on cached DINOv3 features to jointly predict:
- Family (multiclass)
- Order (multiclass)
- Habitat (multi-label over 8 binary attributes: Tropical, Temperate, Subtropical, Boreal, Polar, freshwater, saltwater, brackish)
- Troph (regression)

Notes:
- FeedingPath is treated as a categorical attribute (separate head) if you enable `--include-feedingpath`.
- Features are extracted once via `projects/fishnet2/extract_features.py`.

## 1) Extract features (once)
Use the existing FishNet feature extractor. Example:

```bash
python -m dinov3.projects.fishnet2.extract_features \
  --dataset-root /path/to/datasets/fishnet \
  --ann-root /path/to/datasets/anns \
  --output-dir artifacts/fishnet2_balanced \
  --hub-name dinov3_vitl16 \
  --weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --batch-size 256 --num-workers 8 --fp16
```
This will save `train_features.pt`, `val_features.pt`, `test_features.pt` and `metadata.json` in the output directory.

## 2) Train MMoE

```bash
python -m dinov3.projects.multitask_moe.train_mmoe \
  --features-dir artifacts/fishnet2_balanced \
  --ann-root datasets/anns \
  --output-dir artifacts/mmoe_runs \
  --epochs 40 --batch-size 256 --lr 2e-3 \
  --experts 4 --expert-hidden 2048 --tower-hidden 1024 \
  --loss-weights 1.0 0.5 0.5 0.5 \
  --standardize --wandb
```

Arguments (selected):
- `--features-dir`: Folder with `*_features.pt` and `metadata.json` produced by feature extraction.
- `--ann-root`: Folder containing `train.csv` and `test.csv` with columns described in your dataset.
- `--experts`: Number of experts in the shared expert pool.
- `--loss-weights`: Weights for [family, order, habitat, troph] losses.
- `--include-feedingpath`: Add an optional multi-class head for FeedingPath categories.

Outputs are written to `--output-dir/<auto_exp_name>/` including checkpoints and JSON metrics.

## 3) Evaluate
The script runs validation each epoch and optionally tests if test features are present. It reports per-task metrics:
- Family/Order: accuracy (top-1)
- Habitat: average per-label AUROC (if sklearn installed) or F1@0.5
- Troph: MAE and RMSE

## Tips
- Start with fewer experts (2–4) and moderate hidden sizes to avoid overfitting.
- Enable `--standardize` to z-score features per split using train stats.
- Use `--uncertainty-weighting` to learn loss weights automatically (optional).
