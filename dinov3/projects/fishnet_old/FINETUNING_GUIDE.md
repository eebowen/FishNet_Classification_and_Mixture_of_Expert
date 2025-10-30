# Fine-tuning Guide for FishNet

## 🎯 Overview

This guide covers **two training approaches** for the FishNet dataset with DINOv3:

1. **Linear Probe** (existing): Fast, frozen backbone
2. **Fine-tuning** (new): Slower, trainable backbone, better performance

---

## 📊 Comparison: Linear Probe vs Fine-tuning

| Aspect | Linear Probe | Fine-tuning |
|--------|--------------|-------------|
| **Training Time** | ⚡ Fast (5-15 min) | 🐢 Slower (1-4 hours) |
| **GPU Memory** | 💾 Low (4-6 GB) | 💾 Higher (8-16 GB) |
| **Backbone Weights** | ❄️ Frozen | 🔥 Trainable |
| **Pipeline** | 2-step (extract → train) | 1-step (end-to-end) |
| **Best For** | Quick baselines, many experiments | Maximum accuracy |
| **Data Efficiency** | Good with pretraining | Better with domain shift |
| **Expected Accuracy** | Baseline | +2-10% improvement |

---

## 🚀 Usage

### Option 1: Linear Probe (Existing)

**When to use:**
- Quick experiments
- Limited compute
- Testing different hyperparameters
- DINOv3 pretraining closely matches your domain

**Steps:**

```bash
# Step 1: Extract features (once)
python -m dinov3.projects.fishnet.extract_features \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/features \
  --hub-name dinov3_vits16 \
  --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --batch-size 256 \
  --num-workers 8

# Step 2: Train classifier (fast, can repeat with different settings)
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

**Advantages:**
- ✅ Experiment with many hyperparameters quickly
- ✅ Feature extraction is one-time cost
- ✅ Low memory requirements

---

### Option 2: Fine-tuning (New)

**When to use:**
- Maximum accuracy is priority
- You have compute resources
- Domain shift from ImageNet (fish images are different)
- After finding good hyperparameters with linear probe

**Basic fine-tuning:**

```bash
python -m dinov3.projects.fishnet.train_finetune \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/finetune_vits16 \
  --hub-name dinov3_vits16 \
  --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --epochs 50 \
  --batch-size 64 \
  --lr-backbone 1e-5 \
  --lr-head 1e-3 \
  --class-weighting \
  --use-strong-aug \
  --fp16
```

**Progressive fine-tuning (recommended):**

```bash
# Stage 1: Train head only (faster, like linear probe but online)
python -m dinov3.projects.fishnet.train_finetune \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/finetune_stage1 \
  --hub-name dinov3_vits16 \
  --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --freeze-epochs 50 \
  --epochs 50 \
  --batch-size 128 \
  --lr-head 1e-3

# Stage 2: Fine-tune last few layers
python -m dinov3.projects.fishnet.train_finetune \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/finetune_stage2 \
  --hub-name dinov3_vits16 \
  --weights artifacts/fishnet/finetune_stage1/best_model.pt \
  --unfreeze-layers 4 \
  --epochs 30 \
  --batch-size 64 \
  --lr-backbone 5e-6 \
  --lr-head 1e-4 \
  --use-strong-aug

# Stage 3: Fine-tune entire backbone (optional)
python -m dinov3.projects.fishnet.train_finetune \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/finetune_stage3 \
  --hub-name dinov3_vits16 \
  --weights artifacts/fishnet/finetune_stage2/best_model.pt \
  --unfreeze-layers -1 \
  --epochs 20 \
  --batch-size 32 \
  --lr-backbone 1e-6 \
  --lr-head 5e-5 \
  --weight-decay 0.05 \
  --use-strong-aug
```

---

## 🎛️ Key Hyperparameters for Fine-tuning

### Learning Rates
```bash
--lr-backbone 1e-5   # Very small for pretrained weights
--lr-head 1e-3       # Larger for randomly initialized head
```

**Rules of thumb:**
- Backbone LR: 10-100x smaller than head LR
- Start with `1e-5` for backbone, `1e-3` for head
- If training is unstable: reduce both by 2-5x

### Batch Size
```bash
--batch-size 64  # Smaller than linear probe due to memory
```

- Reduce if out of memory (32 or 16)
- Increase if you have larger GPU (128 or 256)
- Use `--fp16` for mixed precision to fit larger batches

### Freezing Strategy
```bash
--freeze-epochs 10      # Freeze backbone for first 10 epochs
--unfreeze-layers 4     # Only train last 4 transformer blocks
--unfreeze-layers -1    # Train entire backbone
```

**Progressive unfreezing (best practice):**
1. Train head with frozen backbone (fast warmup)
2. Unfreeze last few layers (gradual adaptation)
3. Optionally unfreeze all layers (full fine-tuning)

### Regularization
```bash
--weight-decay 0.05      # L2 regularization
--dropout 0.1            # Dropout in classifier head
--label-smoothing 0.1    # Prevent overconfidence
--use-strong-aug         # Stronger data augmentation
```

**When to increase regularization:**
- Small dataset (<10k images)
- Overfitting (train acc >> val acc)
- Fine-tuning entire backbone

---

## 💡 Best Practices

### 1. **Start with Linear Probe**
```bash
# Quick baseline (15 minutes)
python -m dinov3.projects.fishnet.extract_features ...
python -m dinov3.projects.fishnet.train_linear ...
```

### 2. **If linear probe gives good results (>80% acc):**
- Fine-tuning may only add 2-5% improvement
- Consider if the extra time is worth it

### 3. **If linear probe gives poor results (<60% acc):**
- Domain shift is likely significant
- Fine-tuning should help substantially (+10-20%)

### 4. **Progressive fine-tuning recipe:**
```bash
# Stage 1: Head only (50 epochs, ~1 hour)
--freeze-epochs 50 --lr-head 1e-3

# Stage 2: Last 4 blocks (30 epochs, ~2 hours)
--unfreeze-layers 4 --lr-backbone 5e-6 --lr-head 1e-4

# Stage 3: Full model (20 epochs, ~3 hours) - optional
--unfreeze-layers -1 --lr-backbone 1e-6 --lr-head 5e-5
```

### 5. **Memory optimization:**
```bash
--fp16              # Use mixed precision (saves ~40% memory)
--batch-size 32     # Reduce if needed
--num-workers 4     # Reduce if CPU bottleneck
```

### 6. **Monitor training:**
```bash
# Check for overfitting
tail -f artifacts/fishnet/finetune_vits16/training.log

# Look for:
# - Train acc >> Val acc → increase regularization
# - Both improving → good
# - Val acc plateauing → early stopping working
```

---

## 📈 Expected Results

### Linear Probe (ViT-S/16)
- Training time: ~15 minutes
- Memory: ~6 GB
- Expected accuracy: 60-80% (depends on dataset)

### Fine-tuning (ViT-S/16, progressive)
- Training time: ~4-6 hours total
- Memory: ~12 GB
- Expected accuracy: 70-90% (+5-15% over linear probe)

### Fine-tuning (ViT-L/14, full)
- Training time: ~12-24 hours
- Memory: ~24 GB
- Expected accuracy: 75-95% (best possible)

---

## 🔧 Troubleshooting

### Out of Memory
```bash
# Solutions (try in order):
--fp16                    # Enable mixed precision
--batch-size 16           # Reduce batch size
--unfreeze-layers 2       # Freeze more layers
--gradient-checkpointing  # Trade compute for memory (if implemented)
```

### Training is Unstable (loss → NaN)
```bash
# Solutions:
--lr-backbone 1e-6        # Reduce learning rate
--lr-head 5e-4
--weight-decay 0.01       # Reduce weight decay
--label-smoothing 0.0     # Disable label smoothing
```

### Overfitting (train acc >> val acc)
```bash
# Solutions:
--dropout 0.2             # Increase dropout
--weight-decay 0.1        # Increase weight decay
--use-strong-aug          # Enable stronger augmentation
--freeze-epochs 20        # Keep backbone frozen longer
```

### Underfitting (both acc low)
```bash
# Solutions:
--epochs 100              # Train longer
--lr-backbone 5e-5        # Increase learning rate
--unfreeze-layers -1      # Unfreeze more layers
--hidden-dim 4096         # Larger classifier head
```

---

## 🎓 When to Use Each Approach

### Use Linear Probe When:
- ✅ Doing rapid prototyping
- ✅ Testing many hyperparameters
- ✅ Limited GPU resources
- ✅ Dataset is similar to ImageNet
- ✅ Good enough accuracy (>80%)

### Use Fine-tuning When:
- ✅ Need maximum accuracy
- ✅ Have GPU resources (8+ GB VRAM)
- ✅ Domain shift from ImageNet (medical, satellite, fish, etc.)
- ✅ Linear probe gives <70% accuracy
- ✅ Final production model

### Hybrid Approach (Recommended):
1. **Linear probe first** → establish baseline (15 min)
2. **Analyze results** → check per-class accuracy
3. **If good enough** → done! Use linear probe model
4. **If not** → fine-tune with insights from linear probe

---

## 📚 Additional Resources

- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [Fine-tuning Best Practices](https://huggingface.co/blog/fine-tune-vit)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)

---

## 🤝 Contributing

If you find better hyperparameters or strategies, please share them!

- Open an issue with your results
- Submit a PR with improved configurations
- Document your findings in this guide
