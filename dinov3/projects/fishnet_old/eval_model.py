"""Evaluate a trained linear classifier and report per-class accuracy."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import from train_linear
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_linear import LinearClassifier, load_features, standardize_features

logger = logging.getLogger("dinov3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained classifier with per-class accuracy")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model .pt file")
    parser.add_argument("--test-features", type=Path, required=True, help="Path to test features .pt file")
    parser.add_argument("--train-features", type=Path, default=None, help="Optional: for standardization stats")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to evaluate on",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply standardization using train stats (must provide --train-features)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dimension (must match trained model)",
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict:
    """Evaluate model and compute per-class accuracy."""
    model.eval()
    
    # Collect all predictions and targets
    all_preds = []
    all_targets = []
    
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        preds = logits.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Compute per-class accuracy
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    
    for pred, target in zip(all_preds, all_targets):
        per_class_total[target] += 1
        if pred == target:
            per_class_correct[target] += 1
    
    # Avoid division by zero
    per_class_acc = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros_like(per_class_correct),
        where=per_class_total > 0
    )
    
    # Overall accuracy
    overall_acc = (all_preds == all_targets).mean()
    
    # Compute confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, target in zip(all_preds, all_targets):
        confusion[target, pred] += 1
    
    return {
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc.tolist(),
        "per_class_correct": per_class_correct.tolist(),
        "per_class_total": per_class_total.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata for class names
    metadata = {}
    metadata_path = args.test_features.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        class_names = metadata.get("classes", [])
    else:
        class_names = []
        logger.warning("Metadata not found, class names will be numeric")
    
    # Load test features
    logger.info(f"Loading test features from {args.test_features}")
    test_features, test_labels = load_features(args.test_features)
    
    # Apply standardization if requested
    if args.standardize:
        if args.train_features is None:
            raise ValueError("--train-features required when --standardize is used")
        logger.info("Applying standardization using train statistics")
        train_features, _ = load_features(args.train_features)
        _, test_features = standardize_features(train_features, test_features)
    
    # Determine number of classes
    num_classes = int(test_labels.max().item() + 1)
    in_dim = test_features.size(1)
    
    logger.info(f"Feature dim: {in_dim}, Number of classes: {num_classes}")
    
    # Load model
    device = torch.device(args.device)
    model = LinearClassifier(in_dim, num_classes, args.hidden_dim).to(device)
    
    logger.info(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Create DataLoader
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate_per_class(model, test_loader, device, num_classes)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall Test Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    
    # Per-class accuracy summary
    per_class_acc = np.array(results['per_class_accuracy'])
    per_class_total = np.array(results['per_class_total'])
    
    # Filter out classes not in test set
    valid_classes = per_class_total > 0
    valid_acc = per_class_acc[valid_classes]
    
    print(f"\nPer-Class Accuracy Statistics (over {valid_classes.sum()} classes in test set):")
    print(f"  Mean: {valid_acc.mean():.4f}")
    print(f"  Std:  {valid_acc.std():.4f}")
    print(f"  Min:  {valid_acc.min():.4f}")
    print(f"  Max:  {valid_acc.max():.4f}")
    
    # Show worst performing classes
    print("\n" + "=" * 70)
    print("WORST 20 CLASSES (by accuracy):")
    print("=" * 70)
    
    # Create list of (class_idx, accuracy, count)
    class_performance = []
    for i, (acc, count) in enumerate(zip(per_class_acc, per_class_total)):
        if count > 0:  # Only include classes in test set
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            class_performance.append((i, class_name, acc, int(count)))
    
    # Sort by accuracy
    class_performance.sort(key=lambda x: x[2])
    
    print(f"{'Class ID':<10} {'Class Name':<30} {'Accuracy':<10} {'Test Samples':<15}")
    print("-" * 70)
    for idx, name, acc, count in class_performance[:20]:
        print(f"{idx:<10} {name:<30} {acc:.4f}    {count:<15}")
    
    # Show best performing classes
    print("\n" + "=" * 70)
    print("BEST 20 CLASSES (by accuracy):")
    print("=" * 70)
    print(f"{'Class ID':<10} {'Class Name':<30} {'Accuracy':<10} {'Test Samples':<15}")
    print("-" * 70)
    for idx, name, acc, count in reversed(class_performance[-20:]):
        print(f"{idx:<10} {name:<30} {acc:.4f}    {count:<15}")
    
    print("=" * 70)
    
    # Save detailed results
    detailed_results = {
        "overall_accuracy": results["overall_accuracy"],
        "num_classes": num_classes,
        "num_test_samples": int(test_labels.size(0)),
        "per_class_stats": {
            "mean": float(valid_acc.mean()),
            "std": float(valid_acc.std()),
            "min": float(valid_acc.min()),
            "max": float(valid_acc.max()),
        },
        "per_class_results": []
    }
    
    for idx, name, acc, count in class_performance:
        detailed_results["per_class_results"].append({
            "class_id": idx,
            "class_name": name,
            "accuracy": float(acc),
            "test_samples": count,
            "correct": int(results["per_class_correct"][idx])
        })
    
    # Save to file
    output_file = args.output_dir / "per_class_accuracy.json"
    output_file.write_text(json.dumps(detailed_results, indent=2))
    logger.info(f"Saved detailed results to {output_file}")
    
    # Save confusion matrix
    confusion_file = args.output_dir / "confusion_matrix.npy"
    np.save(confusion_file, np.array(results["confusion_matrix"]))
    logger.info(f"Saved confusion matrix to {confusion_file}")
    
    print(f"\n✅ Full results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
