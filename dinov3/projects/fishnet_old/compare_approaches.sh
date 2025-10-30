#!/bin/bash
# Quick comparison: Linear Probe vs Fine-tuning
# This script runs both approaches and compares results

set -e

DATASET_ROOT="datasets/fishnet"
HUB_NAME="dinov3_vits16"
WEIGHTS="weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

echo "======================================"
echo "FishNet Training Comparison"
echo "======================================"
echo ""

# ========================================
# Option 1: Linear Probe (2-step)
# ========================================
echo "1️⃣  OPTION 1: LINEAR PROBE (2-step pipeline)"
echo "   ⏱️  Expected time: ~15 minutes"
echo "   💾 Memory: ~6 GB"
echo ""

# Step 1: Extract features (run once)
echo "📦 Step 1: Extracting features..."
python -m dinov3.projects.fishnet.extract_features \
  --dataset-root "$DATASET_ROOT" \
  --output-dir artifacts/fishnet/features \
  --hub-name "$HUB_NAME" \
  --weights "$WEIGHTS" \
  --batch-size 256 \
  --num-workers 8

echo ""
echo "✅ Features extracted and cached!"
echo ""

# Step 2: Train linear classifier (fast, repeatable)
echo "🎯 Step 2: Training linear classifier..."
python -m dinov3.projects.fishnet.train_linear \
  --train-features artifacts/fishnet/features/train_features.pt \
  --val-features artifacts/fishnet/features/val_features.pt \
  --test-features artifacts/fishnet/features/test_features.pt \
  --output-dir artifacts/fishnet/linear_vits16 \
  --epochs 50 \
  --lr 5e-4 \
  --class-weighting balanced \
  --standardize

echo ""
echo "✅ Linear probe training complete!"
echo ""

# ========================================
# Option 2: Fine-tuning (1-step)
# ========================================
echo ""
echo "2️⃣  OPTION 2: FINE-TUNING (end-to-end)"
echo "   ⏱️  Expected time: ~4 hours"
echo "   💾 Memory: ~12 GB"
echo ""
echo "⚠️  This will take significantly longer!"
read -p "Continue with fine-tuning? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping fine-tuning. Run manually if needed:"
    echo ""
    echo "python -m dinov3.projects.fishnet.train_finetune \\"
    echo "  --dataset-root $DATASET_ROOT \\"
    echo "  --output-dir artifacts/fishnet/finetune_vits16 \\"
    echo "  --hub-name $HUB_NAME \\"
    echo "  --weights $WEIGHTS \\"
    echo "  --epochs 50 \\"
    echo "  --batch-size 64 \\"
    echo "  --lr-backbone 1e-5 \\"
    echo "  --lr-head 1e-3 \\"
    echo "  --class-weighting \\"
    echo "  --use-strong-aug \\"
    echo "  --fp16"
    echo ""
    exit 0
fi

echo "🚀 Starting fine-tuning..."
python -m dinov3.projects.fishnet.train_finetune \
  --dataset-root "$DATASET_ROOT" \
  --output-dir artifacts/fishnet/finetune_vits16 \
  --hub-name "$HUB_NAME" \
  --weights "$WEIGHTS" \
  --epochs 50 \
  --batch-size 64 \
  --lr-backbone 1e-5 \
  --lr-head 1e-3 \
  --class-weighting \
  --use-strong-aug \
  --fp16

echo ""
echo "✅ Fine-tuning complete!"
echo ""

# ========================================
# Compare Results
# ========================================
echo ""
echo "======================================"
echo "📊 RESULTS COMPARISON"
echo "======================================"
echo ""

# Extract accuracies from results files
LINEAR_RESULTS="artifacts/fishnet/linear_vits16/metrics.json"
FINETUNE_RESULTS="artifacts/fishnet/finetune_vits16/finetune_results.json"

if [ -f "$LINEAR_RESULTS" ]; then
    LINEAR_ACC=$(python -c "import json; print(f\"{json.load(open('$LINEAR_RESULTS'))['test_accuracy']*100:.2f}\")")
    echo "Linear Probe Test Accuracy:  $LINEAR_ACC%"
fi

if [ -f "$FINETUNE_RESULTS" ]; then
    FINETUNE_ACC=$(python -c "import json; print(f\"{json.load(open('$FINETUNE_RESULTS'))['best_val_accuracy']*100:.2f}\")")
    echo "Fine-tuning Val Accuracy:    $FINETUNE_ACC%"
fi

echo ""
echo "======================================"
echo "💡 Recommendations:"
echo "======================================"
echo ""
echo "1. Linear Probe is great for:"
echo "   - Quick experimentation"
echo "   - Hyperparameter tuning"
echo "   - Baseline results"
echo ""
echo "2. Fine-tuning is better when:"
echo "   - Linear probe accuracy < 70%"
echo "   - Maximum performance needed"
echo "   - Domain shift from ImageNet"
echo ""
echo "3. Try progressive fine-tuning:"
echo "   - Stage 1: Freeze backbone (fast)"
echo "   - Stage 2: Unfreeze last layers"
echo "   - Stage 3: Full fine-tuning"
echo ""
echo "See FINETUNING_GUIDE.md for details!"
echo ""
