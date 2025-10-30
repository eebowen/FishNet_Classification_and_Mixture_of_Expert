"""Train a linear classifier on cached DINOv3 features for the FishNet dataset."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["none", "cosine", "step", "plateau", "linear"],
        default="cosine",
        help="Learning rate scheduler type (default: cosine)",
    )
    parser.add_argument("--lr-warmup-epochs", type=int, default=5, help="Number of warmup epochs for lr scheduler")
    parser.add_argument("--lr-step-size", type=int, default=10, help="Step size for StepLR scheduler")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Gamma for StepLR scheduler")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="If no val split provided, ratio of train for validation (set to 0 to use full train set)")
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
        "--no-stratify",
        action="store_true",
        help="Disable stratified train/val split (by default, stratified split is used to preserve class distribution)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided: <scheduler>_lr<lr>_bs<batch>_<timestamp>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (e.g. cuda, cuda:1, cpu)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="fishnet",
        help="Weights & Biases project name (default: fishnet)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username or team name)",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="MixUp alpha for feature-space mixing (0 disables MixUp)",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to feature vectors during training (0 disables noise)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing value for cross-entropy loss (0 to disable)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability to apply MixUp to a training batch (default 1.0)",
    )
    parser.add_argument(
        "--mixup-intra-class",
        action="store_true",
        help="When set, only mix samples within the same class (safer for feature-space MixUp)",
    )
    parser.add_argument(
        "--dropout-prob",
        type=float,
        default=0.2,
        help="Dropout probability used in the linear head (LayerNorm+Dropout head)",
    )

    args = parser.parse_args()
    if args.val_features is None and args.val_ratio > 0 and not (0.0 < args.val_ratio < 0.5):
        parser.error("When --val-features is not provided and --val-ratio > 0, it must be between 0 and 0.5")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be a positive integer")
    if args.mixup_alpha < 0:
        parser.error("--mixup-alpha must be >= 0")
    if not (0.0 <= args.mixup_prob <= 1.0):
        parser.error("--mixup-prob must be in [0, 1]")
    if args.feature_noise_std < 0:
        parser.error("--feature-noise-std must be >= 0")
    if args.label_smoothing < 0 or args.label_smoothing >= 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if not (0.0 <= args.dropout_prob < 1.0):
        parser.error("--dropout-prob must be in [0, 1)")
    return args


def load_features(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    labels = payload["labels"].long().view(-1)
    return features, labels


def split_train_validation(features: torch.Tensor, labels: torch.Tensor, ratio: float, seed: int, stratify: bool = True):
    """
    Split features into train and validation sets.
    
    Args:
        features: Feature tensor
        labels: Label tensor
        ratio: Validation ratio
        seed: Random seed
        stratify: If True, maintain class distribution in both splits (recommended for imbalanced datasets)
    """
    generator = torch.Generator().manual_seed(seed)
    total = features.size(0)
    if total < 2:
        raise ValueError("Need at least two samples to create a validation split")
    
    if not stratify:
        # Random split
        val_size = max(1, int(total * ratio))
        val_size = min(val_size, total - 1)
        indices = torch.randperm(total, generator=generator)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        return (features[train_idx], labels[train_idx]), (features[val_idx], labels[val_idx])
    
    # Stratified split - preserve class distribution
    train_indices = []
    val_indices = []
    
    unique_labels = labels.unique()
    for label in unique_labels:
        label_mask = labels == label
        label_indices = torch.where(label_mask)[0]
        n_samples = len(label_indices)
        
        # For classes with very few samples, ensure at least 1 goes to train
        if n_samples == 1:
            train_indices.extend(label_indices.tolist())
            logger.warning(f"Class {label.item()} has only 1 sample, assigning to train only")
        elif n_samples == 2:
            # With 2 samples, put 1 in train and 1 in val
            perm = torch.randperm(n_samples, generator=generator)
            train_indices.append(label_indices[perm[0]].item())
            val_indices.append(label_indices[perm[1]].item())
        else:
            # Split proportionally, ensuring at least 1 sample in each split
            n_val = max(1, min(n_samples - 1, int(n_samples * ratio)))
            perm = torch.randperm(n_samples, generator=generator)
            shuffled = label_indices[perm]
            val_indices.extend(shuffled[:n_val].tolist())
            train_indices.extend(shuffled[n_val:].tolist())
    
    train_idx = torch.tensor(train_indices)
    val_idx = torch.tensor(val_indices)
    
    logger.info(
        "Stratified split: %d train samples, %d val samples across %d classes",
        len(train_idx), len(val_idx), len(unique_labels)
    )
    
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
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int, dropout_prob: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
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
    mixup_alpha: float = 0.0,
    feature_noise_std: float = 0.0,
    mixup_prob: float = 1.0,
    mixup_intra_class: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        # MixUp in feature space (optional, with probability and intra-class option)
        do_mixup = mixup_alpha and mixup_alpha > 0.0 and (torch.rand(()) <= mixup_prob)
        if do_mixup:
            beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
            lam = float(beta.sample())
            B = features.size(0)
            if mixup_intra_class:
                # Build permutation that pairs samples within the same class
                indices = torch.empty(B, dtype=torch.long, device=device)
                # For each label present in the batch, shuffle indices of that label
                unique = labels.unique()
                for c in unique:
                    mask = (labels == c)
                    idx_c = torch.where(mask)[0]
                    if idx_c.numel() > 1:
                        perm = idx_c[torch.randperm(idx_c.numel(), device=device)]
                    else:
                        perm = idx_c  # only one sample of this class; fallback to self
                    indices[mask] = perm
            else:
                indices = torch.randperm(B, device=device)
            mixed_features = lam * features + (1.0 - lam) * features[indices]
            targets_a = labels
            targets_b = labels[indices]
            features_for_model = mixed_features
        else:
            lam = None
            targets_a = None
            targets_b = None
            features_for_model = features

        # Add Gaussian noise to features if requested
        if feature_noise_std and feature_noise_std > 0.0:
            features_for_model = features_for_model + torch.randn_like(features_for_model) * feature_noise_std

        logits = model(features_for_model)
        if lam is not None:
            # Linear combination of CE losses preserves compatibility with class weights and label smoothing
            loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
        else:
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        if lam is not None:
            # Use the dominant label for accuracy accounting
            hard_targets = targets_a if lam >= 0.5 else targets_b
            total_correct += (preds == hard_targets).sum().item()
        else:
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


def generate_exp_name(args: argparse.Namespace) -> str:
    """Generate experiment name from hyperparameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive name from key hyperparameters
    name_parts = []
    
    # Scheduler info
    if args.lr_scheduler != "none":
        sched_name = args.lr_scheduler
        if args.lr_scheduler == "cosine" and args.lr_warmup_epochs > 0:
            sched_name += f"_w{args.lr_warmup_epochs}"
        name_parts.append(sched_name)
    
    # Learning rate
    lr_str = f"lr{args.lr:.0e}".replace("e-0", "e-").replace("e-", "e")
    name_parts.append(lr_str)
    
    # Batch size
    name_parts.append(f"bs{args.batch_size}")
    
    # Hidden dim if not default
    if args.hidden_dim != 2048:
        name_parts.append(f"h{args.hidden_dim}")
    
    # Class weighting
    if args.class_weighting == "balanced":
        name_parts.append("balanced")
    
    # Standardization
    if args.standardize:
        name_parts.append("std")
    
    # Regularization flags (shortened)
    if getattr(args, "mixup_alpha", 0.0) > 0:
        name_parts.append(f"mx{int(args.mixup_alpha * 100)}")
    if getattr(args, "feature_noise_std", 0.0) > 0:
        name_parts.append(f"nz{int(args.feature_noise_std * 100)}")
    if getattr(args, "label_smoothing", 0.0) > 0:
        name_parts.append(f"ls{int(args.label_smoothing * 100)}")
    if getattr(args, "mixup_prob", 1.0) < 1.0:
        name_parts.append(f"mp{int(args.mixup_prob * 100)}")
    if getattr(args, "mixup_intra_class", False):
        name_parts.append("mic")
    if getattr(args, "dropout_prob", 0.0) > 0.0:
        name_parts.append(f"dp{int(args.dropout_prob * 100)}")
    
    # Timestamp
    name_parts.append(timestamp)
    
    return "_".join(name_parts)


def setup_experiment_dir(base_output_dir: Path, exp_name: str) -> Path:
    """
    Create experiment directory structure and return the experiment path.
    
    Structure:
        base_output_dir/
            exp_name/
                checkpoints/
                logs/
                results/
    """
    exp_dir = base_output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    return exp_dir


def save_experiment_config(output_dir: Path, args: argparse.Namespace) -> None:
    """Save all hyperparameters and experiment configuration."""
    config = vars(args).copy()
    
    # Convert Path objects to strings for JSON serialization
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    logger.info(f"Saved experiment config to {config_path}")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
    step_size: int = 10,
    gamma: float = 0.1,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("none", "cosine", "step", "plateau", "linear")
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        step_size: Step size for StepLR
        gamma: Multiplicative factor for StepLR
        
    Returns:
        Scheduler object or None if scheduler_type is "none"
    """
    if scheduler_type == "none":
        return None
    
    if scheduler_type == "cosine":
        # Cosine annealing with warmup
        if warmup_epochs > 0:
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        logger.info(f"Using CosineAnnealingLR scheduler with {warmup_epochs} warmup epochs")
        
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}")
        
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        logger.info("Using ReduceLROnPlateau scheduler")
        
    elif scheduler_type == "linear":
        # Linear decay from initial LR to near-zero over all epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs
        )
        logger.info(f"Using LinearLR scheduler decaying over {epochs} epochs")
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def main() -> None:
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Check wandb availability
    if args.wandb and not WANDB_AVAILABLE:
        logger.warning("wandb requested but not installed. Install with: pip install wandb")
        args.wandb = False
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        args.exp_name = generate_exp_name(args)
        logger.info(f"Auto-generated experiment name: {args.exp_name}")
    
    # Setup experiment directory structure
    exp_dir = setup_experiment_dir(args.output_dir, args.exp_name)
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Save experiment configuration
    save_experiment_config(exp_dir, args)
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb_config = vars(args).copy()
        # Convert Path objects to strings for wandb
        for key, value in wandb_config.items():
            if isinstance(value, Path):
                wandb_config[key] = str(value)
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=wandb_config,
            dir=str(exp_dir),
        )
        logger.info(f"Initialized wandb tracking for project '{args.wandb_project}'")
    
    # Setup logging to file
    log_file = exp_dir / "logs" / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting experiment: {args.exp_name}")
    logger.info(f"Configuration: {json.dumps(vars(args), indent=2, default=str)}")

    metadata = {}
    metadata_path = args.train_features.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    train_features, train_labels = load_features(args.train_features)
    if args.val_features is not None:
        val_features, val_labels = load_features(args.val_features)
        logger.info("Loaded validation features from %s", args.val_features)
    elif args.val_ratio > 0:
        (train_features, train_labels), (val_features, val_labels) = split_train_validation(
            train_features, train_labels, args.val_ratio, args.seed, stratify=not args.no_stratify
        )
        split_type = "Stratified split" if not args.no_stratify else "Random split"
        logger.info("%s training data: %d train, %d val", split_type, len(train_features), len(val_features))
    else:
        # No validation split - use training data as validation for early stopping
        val_features, val_labels = train_features, train_labels
        logger.info("No validation split specified, using training data for monitoring")

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

    model = LinearClassifier(in_dim, num_classes, args.hidden_dim, dropout_prob=args.dropout_prob).to(device)

    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = compute_class_weights(train_labels).to(device)
        logger.info("Using class-balanced loss weights: %s", [round(x, 4) for x in class_weights.cpu().tolist()])

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type=args.lr_scheduler,
        epochs=args.epochs,
        warmup_epochs=args.lr_warmup_epochs,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    train_loader, val_loader = make_dataloaders((train_features, train_labels), (val_features, val_labels), args.batch_size)

    best_metrics: Dict[str, Dict[str, float]] = {}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_alpha=args.mixup_alpha,
            feature_noise_std=args.feature_noise_std,
            mixup_prob=args.mixup_prob,
            mixup_intra_class=args.mixup_intra_class,
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        
        # Step the scheduler
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_stats["accuracy"])
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            "Epoch %d: train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f lr=%.6f",
            epoch,
            train_stats["loss"],
            train_stats["accuracy"],
            val_stats["loss"],
            val_stats["accuracy"],
            current_lr,
        )
        
        # Log to wandb
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "train/accuracy": train_stats["accuracy"],
                "val/loss": val_stats["loss"],
                "val/accuracy": val_stats["accuracy"],
                "lr": current_lr,
            })

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

    # Save model checkpoint
    checkpoint_path = exp_dir / "checkpoints" / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    logger.info(f"Saved best model to {checkpoint_path}")

    val_eval_loader = DataLoader(
        TensorDataset(val_features, val_labels), batch_size=args.batch_size, shuffle=False
    )
    preds_val, targets_val = collect_predictions(model, val_eval_loader, device)
    confusion_val_path = exp_dir / "results" / "val_confusion.pt"
    torch.save(confusion_matrix(preds_val, targets_val, num_classes), confusion_val_path)

    if args.test_features is not None and test_labels is not None:
        test_loader = DataLoader(
            TensorDataset(test_features, test_labels), batch_size=args.batch_size, shuffle=False
        )
        test_stats = evaluate(model, test_loader, criterion, device)
        best_metrics["test"] = test_stats

        preds_test, targets_test = collect_predictions(model, test_loader, device)
        confusion_test_path = exp_dir / "results" / "test_confusion.pt"
        torch.save(confusion_matrix(preds_test, targets_test, num_classes), confusion_test_path)

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
            report_path = exp_dir / "results" / "test_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            logger.info(f"Saved test classification report to {report_path}")

    # Add training configuration to metrics
    best_metrics["config"] = {
        "experiment_name": args.exp_name,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "class_weighting": args.class_weighting,
        "standardize": args.standardize,
        "stratify": not args.no_stratify,
        "seed": args.seed,
        "mixup_alpha": args.mixup_alpha,
        "feature_noise_std": args.feature_noise_std,
        "label_smoothing": args.label_smoothing,
        "mixup_prob": args.mixup_prob,
        "mixup_intra_class": args.mixup_intra_class,
        "dropout_prob": args.dropout_prob,
    }

    metrics_path = exp_dir / "results" / "metrics.json"
    save_metrics(metrics_path, best_metrics)
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info("Finished training. Best metrics: %s", json.dumps(best_metrics, indent=2))
    
    # Log final metrics to wandb
    if args.wandb:
        wandb_summary = {
            "best_val_accuracy": best_metrics["val"]["accuracy"],
            "best_val_loss": best_metrics["val"]["loss"],
            "best_train_accuracy": best_metrics["train"]["accuracy"],
            "best_train_loss": best_metrics["train"]["loss"],
        }
        if "test" in best_metrics:
            wandb_summary["test_accuracy"] = best_metrics["test"]["accuracy"]
            wandb_summary["test_loss"] = best_metrics["test"]["loss"]
        
        wandb.log(wandb_summary)
        
        # Save confusion matrices as wandb artifacts
        if (exp_dir / "results" / "val_confusion.pt").exists():
            wandb.save(str(exp_dir / "results" / "val_confusion.pt"))
        if (exp_dir / "results" / "test_confusion.pt").exists():
            wandb.save(str(exp_dir / "results" / "test_confusion.pt"))
        if (exp_dir / "results" / "test_report.json").exists():
            wandb.save(str(exp_dir / "results" / "test_report.json"))
        
        wandb.finish()
        logger.info("Finished wandb tracking")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETED: {args.exp_name}")
    print("=" * 80)
    print(f"Output directory: {exp_dir}")
    print(f"Best validation accuracy: {best_metrics['val']['accuracy']:.4f}")
    if 'test' in best_metrics:
        print(f"Test accuracy: {best_metrics['test']['accuracy']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
