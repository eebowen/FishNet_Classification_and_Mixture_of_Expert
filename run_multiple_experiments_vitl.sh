#!/bin/bash

# Add current directory to PYTHONPATH so dinov3 module can be found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

BASE_OUTPUT="artifacts/experiments/vit_l"
FEATURES="artifacts/features/vit_l"

# Feature extraction (if not already done)
python dinov3/projects/fishnet2/extract_features.py \
    --dataset-root datasets/fishnet \
    --ann-root datasets/anns \
    --output-dir $FEATURES \
    --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --val-ratio 0.15 \
    --seed 17


# Run multiple experiments with different hyperparameters
# Each will be automatically organized in its own subdirectory
echo "Running Experiment 1: Baseline with Cosine Scheduler"
python dinov3/projects/fishnet2/train_linear.py \
    --train-features $FEATURES/train_features.pt \
    --val-features $FEATURES/val_features.pt \
    --test-features $FEATURES/test_features.pt \
    --output-dir $BASE_OUTPUT \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --patience 30 \
    --class-weighting balanced \
    --standardize \
    --lr-scheduler cosine \
    --lr-warmup-epochs 5 \
    --wandb \
    --wandb-project fishnet

echo -e "\n\nRunning Experiment 2: Higher Learning Rate"
python dinov3/projects/fishnet2/train_linear.py \
    --train-features $FEATURES/train_features.pt \
    --val-features $FEATURES/val_features.pt \
    --test-features $FEATURES/test_features.pt \
    --output-dir $BASE_OUTPUT \
    --epochs 100 \
    --batch-size 256 \
    --lr 5e-3 \
    --patience 30 \
    --class-weighting balanced \
    --standardize \
    --lr-scheduler cosine \
    --lr-warmup-epochs 5 \
    --wandb \
    --wandb-project fishnet

echo -e "\n\nRunning Experiment 3: Step Scheduler"
python dinov3/projects/fishnet2/train_linear.py \
    --train-features $FEATURES/train_features.pt \
    --val-features $FEATURES/val_features.pt \
    --test-features $FEATURES/test_features.pt \
    --output-dir $BASE_OUTPUT \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --patience 30 \
    --class-weighting balanced \
    --standardize \
    --lr-scheduler step \
    --lr-step-size 20 \
    --wandb \
    --wandb-project fishnet


echo -e "\n\nRunning Experiment 4: Larger Batch Size"
python dinov3/projects/fishnet2/train_linear.py \
    --train-features $FEATURES/train_features.pt \
    --val-features $FEATURES/val_features.pt \
    --test-features $FEATURES/test_features.pt \
    --output-dir $BASE_OUTPUT \
    --epochs 100 \
    --batch-size 512 \
    --lr 2e-3 \
    --patience 30 \
    --class-weighting balanced \
    --standardize \
    --lr-scheduler cosine \
    --lr-warmup-epochs 5 \
    --wandb \
    --wandb-project fishnet


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
    --patience 30 \
    --class-weighting balanced \
    --standardize \
    --lr-warmup-epochs 5


echo -e "\n\n=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Compare results with:"
echo "python compare_experiments.py $BASE_OUTPUT"


python compare_experiments.py artifacts/experiments