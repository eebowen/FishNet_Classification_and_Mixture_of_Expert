"""Fine-tune DINOv3 backbone + linear classifier on FishNet dataset."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
    make_classification_train_transform,
)

logger = logging.getLogger("dinov3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 backbone on FishNet")
    
    # Dataset arguments
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory of FishNet dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs")
    
    # Model arguments
    parser.add_argument(
        "--hub-name",
        type=str,
        default="dinov3_vits16",
        help="Name of DINOv3 backbone (dinov3_vits16, dinov3_vitb16, etc.)",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to pretrained DINOv3 checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension for classifier head")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (smaller than linear probe)")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers")
    
    # Learning rate arguments
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for backbone (lower)")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for classifier head (higher)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    
    # Fine-tuning strategy
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="Freeze backbone for first N epochs (0 = fine-tune from start)",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=-1,
        help="Number of last transformer blocks to unfreeze (-1 = unfreeze all)",
    )
    
    # Regularization
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for classifier head")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup alpha (0 = disabled)")
    
    # Data arguments
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--class-weighting", action="store_true", help="Use class-balanced loss weighting")
    
    # Augmentation
    parser.add_argument("--resize-size", type=int, default=RESIZE_DEFAULT_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_DEFAULT_SIZE)
    parser.add_argument("--use-strong-aug", action="store_true", help="Use stronger data augmentation")
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    return parser.parse_args()


class FinetuneClassifier(nn.Module):
    """Classifier head with dropout for fine-tuning."""
    
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ModelWithHead(nn.Module):
    """Combines DINOv3 backbone with classification head."""
    
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get CLS token from backbone
        features = self.backbone(x)
        # Apply classifier head
        return self.head(features)


def load_backbone(hub_name: str, weights_path: Path, device: torch.device) -> Tuple[nn.Module, int]:
    """Load pretrained DINOv3 backbone."""
    logger.info(f"Loading backbone: {hub_name} from {weights_path}")
    
    # Load model from torch.hub
    model = torch.hub.load("facebookresearch/dinov3", hub_name, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    
    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        features = model(dummy_input)
        feature_dim = features.shape[-1]
    
    logger.info(f"Backbone loaded. Feature dimension: {feature_dim}")
    return model, feature_dim


def freeze_backbone_partially(model: nn.Module, num_layers_to_unfreeze: int = -1):
    """
    Freeze backbone parameters, optionally unfreezing last N transformer blocks.
    
    Args:
        model: The backbone model
        num_layers_to_unfreeze: Number of last blocks to keep trainable (-1 = unfreeze all)
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    if num_layers_to_unfreeze == -1:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Backbone fully trainable")
    elif num_layers_to_unfreeze > 0:
        # Unfreeze last N blocks
        if hasattr(model, 'blocks'):
            total_blocks = len(model.blocks)
            start_idx = max(0, total_blocks - num_layers_to_unfreeze)
            for i in range(start_idx, total_blocks):
                for param in model.blocks[i].parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze last {num_layers_to_unfreeze} blocks (blocks {start_idx}-{total_blocks-1})")
        else:
            logger.warning("Model doesn't have 'blocks' attribute, unfreezing all")
            for param in model.parameters():
                param.requires_grad = True
    else:
        logger.info("Backbone frozen")


def create_dataloaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    crop_size: int,
    resize_size: int,
    use_strong_aug: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train and validation dataloaders from image folder."""
    
    # Training transform with augmentation
    train_transform = make_classification_train_transform(
        crop_size=crop_size,
        interpolation_mode="bicubic",
        scale_range=(0.08, 1.0) if use_strong_aug else (0.5, 1.0),
        hflip_prob=0.5,
        auto_augment_policy="ta_wide" if use_strong_aug else None,
    )
    
    # Validation transform (no augmentation)
    val_transform = make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation_mode="bicubic",
    )
    
    # Load full dataset
    full_dataset = ImageFolder(root=dataset_root)
    num_classes = len(full_dataset.classes)
    logger.info(f"Dataset: {len(full_dataset)} images, {num_classes} classes")
    
    # Split into train/val
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    train_indices, val_indices = random_split(
        range(len(full_dataset)), 
        [train_size, val_size],
        generator=generator
    )
    
    # Create datasets with different transforms
    train_dataset = ImageFolder(root=dataset_root, transform=train_transform)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    
    val_dataset = ImageFolder(root=dataset_root, transform=val_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    return train_loader, val_loader, num_classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100.0 * total_correct / total:.2f}%"
            )
    
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
    }


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Arguments: {args}")
    
    device = torch.device(args.device)
    
    # Load data
    train_loader, val_loader, num_classes = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
        use_strong_aug=args.use_strong_aug,
    )
    
    # Load backbone
    backbone, feature_dim = load_backbone(args.hub_name, args.weights, device)
    
    # Create classifier head
    head = FinetuneClassifier(
        in_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    
    # Combine into full model
    model = ModelWithHead(backbone, head)
    
    # Freeze backbone if requested
    if args.freeze_epochs > 0:
        freeze_backbone_partially(backbone, num_layers_to_unfreeze=0)
        logger.info(f"Starting with frozen backbone for {args.freeze_epochs} epochs")
    else:
        freeze_backbone_partially(backbone, num_layers_to_unfreeze=args.unfreeze_layers)
    
    # Setup optimizer with different learning rates for backbone and head
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    
    if backbone_params:
        param_groups = [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ]
        logger.info(f"Using two learning rates: backbone={args.lr_backbone}, head={args.lr_head}")
    else:
        param_groups = [{"params": head_params, "lr": args.lr_head}]
        logger.info(f"Backbone frozen, using head lr={args.lr_head}")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        else:
            progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    if args.class_weighting:
        # Compute class weights from training data
        logger.info("Computing class weights...")
        class_counts = torch.zeros(num_classes)
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        weights = class_counts.sum() / (class_counts.clamp_min(1.0) * num_classes)
        weights = weights.to(device)
        logger.info(f"Class weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    else:
        weights = None
    
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=args.label_smoothing,
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 15
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            logger.info("Unfreezing backbone!")
            freeze_backbone_partially(backbone, num_layers_to_unfreeze=args.unfreeze_layers)
            
            # Recreate optimizer with backbone parameters
            backbone_params = [p for p in backbone.parameters() if p.requires_grad]
            param_groups = [
                {"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params, "lr": args.lr_head},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp=args.fp16
        )
        scheduler.step()
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Log
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']*100:.2f}%"
        )
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}%"
        )
        
        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }
            torch.save(checkpoint, args.output_dir / "best_model.pt")
            logger.info(f"✓ Saved new best model (val_acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{patience_limit})")
        
        # Early stopping
        if patience_counter >= patience_limit:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save final results
    results = {
        "best_val_accuracy": float(best_val_acc),
        "final_train_accuracy": float(history["train_acc"][-1]),
        "final_val_accuracy": float(history["val_acc"][-1]),
        "epochs_trained": epoch,
        "history": history,
    }
    
    with open(args.output_dir / "finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
