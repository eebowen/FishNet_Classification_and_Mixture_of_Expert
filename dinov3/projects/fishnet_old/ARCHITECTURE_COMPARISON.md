# Pipeline Architecture Comparison

## Current: Linear Probe (2-Step Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                    STEP 1: EXTRACT FEATURES                 │
│                         (Run Once)                          │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────┐     ┌──────────┐     ┌────────────┐
    │   Images   │────▶│  DINOv3  │────▶│  Features  │
    │            │     │ (Frozen) │     │   (.pt)    │
    └────────────┘     └──────────┘     └────────────┘
                                              │
                                              │ Cache to disk
                                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: TRAIN LINEAR CLASSIFIER                │
│                  (Fast & Repeatable)                        │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────┐     ┌──────────┐     ┌────────────┐
    │  Features  │────▶│  Linear  │────▶│   Model    │
    │   (.pt)    │     │   Head   │     │            │
    └────────────┘     └──────────┘     └────────────┘

✅ Pros:
  - Fast experimentation (Step 2 takes minutes)
  - Low memory (~6 GB)
  - Easy hyperparameter tuning
  - Features cached and reusable

⚠️  Cons:
  - Backbone stays frozen
  - Can't use data augmentation in Step 2
  - Limited improvement potential
```

---

## New: Fine-tuning (End-to-End Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│              END-TO-END TRAINING (One Step)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────┐     ┌──────────┐     ┌──────────┐     ┌───────┐
    │   Images   │────▶│  DINOv3  │────▶│  Linear  │────▶│ Model │
    │    +Aug    │     │Trainable │     │   Head   │     │       │
    └────────────┘     └──────────┘     └──────────┘     └───────┘
                             │                │
                             └────────┬───────┘
                                      │
                              Gradient flows back
                              through entire model

✅ Pros:
  - Maximum accuracy potential
  - Adapts to domain-specific features
  - Can use data augmentation
  - Better for domain shift

⚠️  Cons:
  - Slower training (hours vs minutes)
  - Higher memory (~12 GB)
  - Need to retrain for each experiment
  - Risk of overfitting
```

---

## Hybrid: Progressive Fine-tuning (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│           STAGE 1: Train Head Only (10-20 epochs)           │
│                    Like linear probe                        │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────┐     ┌──────────┐     ┌──────────┐
    │   Images   │────▶│  DINOv3  │────▶│  Linear  │
    │            │     │ ❄️ FROZEN │     │ 🔥 TRAIN │
    └────────────┘     └──────────┘     └──────────┘

┌─────────────────────────────────────────────────────────────┐
│     STAGE 2: Unfreeze Last Layers (20-30 epochs)            │
│              Gradual adaptation                             │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────┐     ┌──────────┐     ┌──────────┐
    │  Images+   │────▶│  DINOv3  │────▶│  Linear  │
    │   Aug      │     │ 🔥 Last 4│     │ 🔥 TRAIN │
    └────────────┘     │  blocks  │     └──────────┘
                       └──────────┘

┌─────────────────────────────────────────────────────────────┐
│      STAGE 3: Full Fine-tuning (10-20 epochs) [Optional]    │
│                Maximum performance                          │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌────────────┐     ┌──────────┐     ┌──────────┐
    │  Images+   │────▶│  DINOv3  │────▶│  Linear  │
    │Strong Aug  │     │🔥 ALL    │     │ 🔥 TRAIN │
    └────────────┘     │  layers  │     └──────────┘
                       └──────────┘

✅ Best of both worlds:
  - Start fast (like linear probe)
  - Progressively improve
  - Reduces overfitting risk
  - Better final performance
```

---

## Decision Tree: Which Approach to Use?

```
                    Start Here
                        │
                        ▼
            ┌───────────────────────┐
            │  Do you have time &   │
            │   GPU resources?      │
            └───────────┬───────────┘
                        │
            ┌───────────┴──────────┐
            │                      │
           NO                     YES
            │                      │
            ▼                      ▼
    ┌──────────────┐      ┌──────────────┐
    │ LINEAR PROBE │      │  Run both!   │
    │              │      │ Start with   │
    │ Fast baseline│      │ Linear Probe │
    └──────────────┘      └──────┬───────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Is accuracy < 70%      │
                    │ or need more?          │
                    └──────┬─────────────────┘
                           │
                   ┌───────┴───────┐
                   │               │
                  NO              YES
                   │               │
                   ▼               ▼
            ┌──────────┐    ┌─────────────┐
            │  Done!   │    │ FINE-TUNE   │
            │ Use      │    │             │
            │ linear   │    │ Progressive │
            │ model    │    │ approach    │
            └──────────┘    └─────────────┘
```

---

## Performance Expectations

### Linear Probe 
```
Training Time:  ███░░░░░░░ (15 min)
GPU Memory:     ████░░░░░░ (6 GB)
Accuracy:       ███████░░░ (Baseline)
Flexibility:    ██████████ (High)
```

### Fine-tuning (Progressive)
```
Training Time:  █████████░ (4-6 hours)
GPU Memory:     ████████░░ (12 GB)
Accuracy:       █████████░ (+5-15%)
Flexibility:    ████░░░░░░ (Lower)
```

### Fine-tuning (Full)
```
Training Time:  ██████████ (8-12 hours)
GPU Memory:     ██████████ (16 GB)
Accuracy:       ██████████ (Maximum)
Flexibility:    ██░░░░░░░░ (Lowest)
```

---

## Code Examples

### Linear Probe (Current)
```bash
# Extract once
python -m dinov3.projects.fishnet.extract_features \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/features \
  --hub-name dinov3_vits16 \
  --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# Train many times with different settings
python -m dinov3.projects.fishnet.train_linear \
  --train-features artifacts/fishnet/features/train_features.pt \
  --val-features artifacts/fishnet/features/val_features.pt \
  --output-dir artifacts/fishnet/linear_vits16 \
  --lr 5e-4 --epochs 50
```

### Fine-tuning (New)
```bash
# Single command, but longer
python -m dinov3.projects.fishnet.train_finetune \
  --dataset-root datasets/fishnet \
  --output-dir artifacts/fishnet/finetune_vits16 \
  --hub-name dinov3_vits16 \
  --weights weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --epochs 50 \
  --lr-backbone 1e-5 \
  --lr-head 1e-3 \
  --fp16
```

---

## Summary Table

| Feature | Linear Probe | Fine-tuning | Progressive |
|---------|-------------|-------------|-------------|
| Pipeline | 2-step | 1-step | 3-stage |
| Time | 15 min | 4-6 hrs | 4-6 hrs |
| Memory | 6 GB | 12 GB | 8-12 GB |
| Accuracy | Base | +5-15% | +5-15% |
| When to use | Baseline | Max perf | Best practice |
| Difficulty | ⭐ Easy | ⭐⭐⭐ Hard | ⭐⭐ Medium |

**Recommendation:** Start with **Linear Probe**, then use **Progressive Fine-tuning** if needed.
