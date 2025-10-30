#!/usr/bin/env python3
"""Quick test to verify CSV data loading works correctly."""
import sys
from pathlib import Path

# Add dinov3 to path
sys.path.insert(0, str(Path(__file__).parent))

from dinov3.projects.fishnet2.extract_features import CSVImageDataset

def test_csv_loading():
    dataset_root = Path("/home/bowen68/projects/fish/dinov3/datasets/fishnet")
    ann_root = Path("/home/bowen68/projects/fish/dinov3/datasets/anns")
    
    train_csv = ann_root / "train.csv"
    test_csv = ann_root / "test.csv"
    
    print("Testing CSV data loading...")
    print(f"Dataset root: {dataset_root}")
    print(f"Annotation root: {ann_root}")
    print()
    
    # Test train dataset
    print("Loading train.csv...")
    train_dataset = CSVImageDataset(train_csv, dataset_root, transform=None)
    print(f"✓ Loaded {len(train_dataset)} training samples")
    print(f"  Number of classes: {len(train_dataset.classes)}")
    
    # Test test dataset
    print("\nLoading test.csv...")
    test_dataset = CSVImageDataset(test_csv, dataset_root, transform=None, class_to_idx=train_dataset.class_to_idx)
    print(f"✓ Loaded {len(test_dataset)} test samples")
    print(f"  Number of classes: {len(test_dataset.classes)}")
    
    # Verify we can load a few samples
    print("\nTesting sample loading...")
    for i in [0, 100, 1000]:
        if i < len(train_dataset):
            img, label = train_dataset[i]
            print(f"  Sample {i}: image size={img.size}, label={label} (class={train_dataset.classes[label]})")
    
    print("\n✓ All tests passed!")
    print(f"\nSummary:")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Total samples: {len(train_dataset) + len(test_dataset):,}")
    print(f"  Number of classes: {len(train_dataset.classes)}")

if __name__ == "__main__":
    test_csv_loading()
