# FishNet Linear Evaluation with DINOv3

This mini-project turns the [`datasets/fishnet`](../../../datasets/fishnet) image taxonomy into a linear-probe benchmark using DINOv3 backbones. It provides two ready-to-run utilities:

1. **`extract_features.py`** – caches frozen DINOv3 embeddings for train/val/test splits.
2. **`train_linear.py`** – fits and evaluates a linear classifier on the cached features.

The workflow is GPU-friendly (enabling distributed collection automatically when CUDA is available) but also supports CPU-only extraction in a pinch.

---

## 1. Environment & prerequisites

1. Create the project environment (CUDA-enabled PyTorch recommended):
   ```bash
   conda env create -f conda.yaml
   conda activate dinov3
   ```
2. Download a DINOv3 backbone checkpoint into the `weights/` directory. For example, use the ViT-S/16 distilled weights (`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`).

> **Tip:** All scripts are runnable via `python -m dinov3.projects.fishnet.<script> ...` from the repository root.

---

## 2. Dataset layout

`extract_features.py` expects the fish imagery to live under `datasets/fishnet/` with class-specific folders. Two layouts are supported:

```
# Explicit splits
fishnet/
  train/<class_name>/<image>.jpg
  val/<class_name>/<image>.jpg
  test/<class_name>/<image>.jpg
```

Or, if no `train/val/test` directories exist, the script will **randomly split** the root tree according to `--val-ratio` and `--test-ratio` (default 15%/15%) using a reproducible seed.

---

## 3. Feature extraction

Caches CLS-token embeddings (L2-normalised by default) for each split.

```bash
python -m dinov3.projects.fishnet.extract_features \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/features \
  --hub-name dinov3_vits16 \
  --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --batch-size 256 \
  --num-workers 8
```

Key notes:
- Distributed collection is automatically enabled when CUDA is available; use `--distributed` explicitly when launching via `torchrun`.
- Add `--fp16` and/or `--multi-scale` for faster/larger throughput.
- Outputs include `train_features.pt`, `val_features.pt`, `test_features.pt`, plus a `metadata.json` describing classes and transforms.

---

## 4. Linear evaluation

Once features are cached:

```bash
python -m dinov3.projects.fishnet.train_linear \
  --train-features artifacts/fishnet/features/train_features.pt \
  --val-features artifacts/fishnet/features/val_features.pt \
  --test-features artifacts/fishnet/features/test_features.pt \
  --output-dir artifacts/fishnet/linear_vits16 \
  --epochs 50 \
  --lr 5e-4 \
  --class-weighting balanced \
  --standardize
```

Tunable knobs worth trying:

- `--hidden-dim`: size of the intermediate layer in the now **two-layer** classifier (defaults to 2048).
- `--val-ratio` / `--test-ratio`: adjust when carving splits from a monolithic dataset.

What you get in `output-dir`:
- `linear_head.pt` – the best-performing head weights.
- `metrics.json` – train/val/test losses & accuracies.
- `val_confusion.pt`, `test_confusion.pt` – integer confusion matrices.
- `test_report.json` (if `scikit-learn` is installed) – full per-class precision/recall metrics.

---

## 5. Troubleshooting & tips

- **Checkpoint mismatch:** Verify the `--hub-name` and `--weights` pair; the ViT-S/16 distilled weights ship with hash `08c60483`.
- **Class imbalance:** Add `--class-weighting balanced` to reweight the cross-entropy loss.
- **No validation split:** Omit `--val-features` to automatically carve one out of the training embeddings (`--val-ratio` controls the portion).
- **CPU-only extraction:** When CUDA is absent, the extractor falls back to a single-process DataLoader (expect slower throughput).
- **More adapters:** To fine-tune adapters or explore other backbones (ConvNeXt, ViT-L/16, etc.), simply adjust the `--hub-name`/`--weights` pair and rerun both scripts.

---
