"""Multi-task Multi-Head (no experts) baseline for FishNet.

This project trains separate two-layer heads per task directly on cached DINOv3
features, to compare against the MMoE project.

Tasks:
- Family (multiclass)
- Order (multiclass)
- Habitat (multi-label over 8 binary attributes)
- Troph (regression)
- Optional: FeedingPath (categorical) via --include-feedingpath
"""
