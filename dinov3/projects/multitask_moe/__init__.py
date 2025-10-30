"""Multi-task Mixture-of-Experts (MMoE) project for FishNet.

This project trains a shared-expert, multi-head model over cached DINOv3 features
for four tasks:
- Family (multiclass)
- Order (multiclass)
- Habitat (multi-label over 8 binary attributes)
- Troph (regression)

Usage overview:
1) Extract features with projects.fishnet2.extract_features
2) Train with projects.multitask_moe.train_mmoe
"""
