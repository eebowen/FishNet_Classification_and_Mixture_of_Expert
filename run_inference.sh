python -m dinov3.projects.multitask_moe.inference_mmoe \
  --checkpoint artifacts/mmoe_runs/mmoe_e8_lr1e-3_bs256_20251028_161524/checkpoints/best_model.pt \
  --features-dir artifacts/fishnet2_balanced/features \
  --dataset-root datasets/fishnet \
  --ann-csv datasets/anns/test.csv \
  --output-dir artifacts/mmoe_visualizations \
  --n-classes 20 \
  --images-per-class 3 \
  --standardize \
  --seed 42


