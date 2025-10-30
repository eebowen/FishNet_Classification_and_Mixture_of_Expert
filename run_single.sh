# python -m dinov3.projects.fishnet.extract_features \
#   --dataset-root datasets/fishnet \
#   --output-dir artifacts/fishnet/features \
#   --hub-name dinov3_vits16 \
#   --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
#   --batch-size 256 \
#   --num-workers 8


# python -m dinov3.projects.fishnet.train_linear \
#   --train-features artifacts/fishnet/features/train_features.pt \
#   --val-features artifacts/fishnet/features/val_features.pt \
#   --test-features artifacts/fishnet/features/test_features.pt \
#   --output-dir artifacts/fishnet/linear_vits16 \
#   --epochs 300 \
#   --lr 5e-4 \
#   --class-weighting balanced \
#   --standardize


# python3 -m dinov3.projects.fishnet.eval_model \
#   --model-path artifacts/fishnet/linear_vits16/linear_head.pt \
#   --test-features artifacts/fishnet/features/test_features.pt \
#   --train-features artifacts/fishnet/features/train_features.pt \
#   --output-dir artifacts/fishnet/linear_vits16/evaluation \
#   --hidden-dim 2048 \
#   --standardize



python dinov3/projects/fishnet2/train_linear.py \
    --train-features artifacts/fishnet2_balanced/features/train_features.pt \
    --val-features artifacts/fishnet2_balanced/features/val_features.pt \
    --test-features artifacts/fishnet2_balanced/features/test_features.pt \
    --output-dir artifacts/experiments \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --lr-scheduler linear \
    --wandb \
    --wandb-project fishnet \
    --patience 50 \
    --class-weighting balanced \
    --standardize \
    --lr-warmup-epochs 5 \
    --dropout-prob 0.1 \
    # --feature-noise-std 0.02 \
    # --label-smoothing 0.05 \
    # --mixup-alpha 0.2 \
    # --mixup-prob 0.5 \
    # --mixup-intra-class \

# Optional safer MixUp variant (uncomment the three lines below and set alpha > 0)
#     --mixup-alpha 0.2 \
#     --mixup-prob 0.5 \
#     --mixup-intra-class \


