"""Train a linear classifier on cached DINOv3 features for the FishNet dataset."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:  # Optional but recommended for richer reports
    from sklearn.metrics import classification_report as sklearn_classification_report
except ImportError:  # pragma: no cover - optional dependency
    sklearn_classification_report = None  # type: ignore

logger = logging.getLogger("dinov3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear classifier on DINOv3 features")
    parser.add_argument("--train-features", type=Path, required=True, help="Path to training features .pt file")
    parser.add_argument("--val-features", type=Path, default=None, help="Optional validation features .pt file")
    parser.add_argument("--test-features", type=Path, default=None, help="Optional test features .pt file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where checkpoints and logs go")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience based on val accuracy")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="If no val split, ratio of train for validation")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dimension of the intermediate layer in the linear classifier",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced"),
        default="none",
        help="Apply class-balanced weighting in the cross-entropy loss",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize (z-score) features using train statistics before training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (e.g. cuda, cuda:1, cpu)",
    )

    args = parser.parse_args()
    if args.val_features is None and not (0.0 < args.val_ratio < 0.5):
        parser.error("When --val-features is not provided, --val-ratio must be between 0 and 0.5")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be a positive integer")
    return args


def load_features(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    labels = payload["labels"].long().view(-1)
    return features, labels


def split_train_validation(features: torch.Tensor, labels: torch.Tensor, ratio: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    total = features.size(0)
    if total < 2:
        raise ValueError("Need at least two samples to create a validation split")
    val_size = max(1, int(total * ratio))
    val_size = min(val_size, total - 1)
    indices = torch.randperm(total, generator=generator)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return (features[train_idx], labels[train_idx]), (features[val_idx], labels[val_idx])


def standardize_features(train_features: torch.Tensor, *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    normalized = [(train_features - mean) / std]
    for feat in others:
        normalized.append((feat - mean) / std)
    return tuple(normalized)


def make_dataloaders(
    train: Tuple[torch.Tensor, torch.Tensor],
    val: Tuple[torch.Tensor, torch.Tensor] | None,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader | None]:
    train_loader = DataLoader(
        TensorDataset(*train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = None
    if val is not None:
        val_loader = DataLoader(
            TensorDataset(*val),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    return train_loader, val_loader


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.layers(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": total_correct / total}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    num_classes = labels.max().item() + 1
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * num_classes)
    return weights


@torch.inference_mode()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, targets = [], []
    for features, labels in loader:
        features = features.to(device)
        logits = model(features)
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels.clone())
    return torch.cat(preds), torch.cat(targets)


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    flat = targets.to(torch.int64) * num_classes + preds.to(torch.int64)
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def save_metrics(path: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    args = parse_args()
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    metadata_path = args.train_features.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    train_features, train_labels = load_features(args.train_features)
    if args.val_features is not None:
        val_features, val_labels = load_features(args.val_features)
    else:
        (train_features, train_labels), (val_features, val_labels) = split_train_validation(
            train_features, train_labels, args.val_ratio, args.seed
        )

    test_features = test_labels = None
    if args.test_features is not None:
        test_features, test_labels = load_features(args.test_features)

    if args.standardize:
        extra = [val_features]
        if test_features is not None:
            extra.append(test_features)
        normalized = standardize_features(train_features, *extra)
        train_features = normalized[0]
        val_features = normalized[1]
        if test_features is not None:
            test_features = normalized[2]

    device = torch.device(args.device)
    in_dim = train_features.size(1)
    num_classes = int(train_labels.max().item() + 1)

    model = LinearClassifier(in_dim, num_classes, args.hidden_dim).to(device)

    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = compute_class_weights(train_labels).to(device)
        logger.info("Using class-balanced loss weights: %s", [round(x, 4) for x in class_weights.cpu().tolist()])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, val_loader = make_dataloaders((train_features, train_labels), (val_features, val_labels), args.batch_size)

    best_metrics: Dict[str, Dict[str, float]] = {}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d: train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
            epoch,
            train_stats["loss"],
            train_stats["accuracy"],
            val_stats["loss"],
            val_stats["accuracy"],
        )

        if val_stats["accuracy"] > best_val + 1e-6:
            best_val = val_stats["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics["train"] = train_stats
            best_metrics["val"] = val_stats
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered after %d epochs", epoch)
                break

    model.load_state_dict(best_state)

    if "train" not in best_metrics:
        best_metrics["train"] = evaluate(model, train_loader, criterion, device)
    if "val" not in best_metrics:
        best_metrics["val"] = evaluate(model, val_loader, criterion, device)

    torch.save(best_state, args.output_dir / "linear_head.pt")

    val_eval_loader = DataLoader(
        TensorDataset(val_features, val_labels), batch_size=args.batch_size, shuffle=False
    )
    preds_val, targets_val = collect_predictions(model, val_eval_loader, device)
    torch.save(confusion_matrix(preds_val, targets_val, num_classes), args.output_dir / "val_confusion.pt")

    if args.test_features is not None and test_labels is not None:
        test_loader = DataLoader(
            TensorDataset(test_features, test_labels), batch_size=args.batch_size, shuffle=False
        )
        test_stats = evaluate(model, test_loader, criterion, device)
        best_metrics["test"] = test_stats

        preds_test, targets_test = collect_predictions(model, test_loader, device)
        torch.save(confusion_matrix(preds_test, targets_test, num_classes), args.output_dir / "test_confusion.pt")

        if sklearn_classification_report is not None:
            target_names = metadata.get("classes") if metadata else None
            labels = None
            if target_names is not None:
                labels = list(range(len(target_names)))
            report = sklearn_classification_report(
                targets_test.numpy(),
                preds_test.numpy(),
                target_names=target_names,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            (args.output_dir / "test_report.json").write_text(json.dumps(report, indent=2))

    save_metrics(args.output_dir / "metrics.json", best_metrics)
    logger.info("Finished training. Metrics: %s", json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
