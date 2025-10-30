from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

from .models import GatedMMoE, TaskSpec
from .data import HABITAT_BINARY_COLS, build_label_encoders, build_split_labels

logger = logging.getLogger("dinov3")


# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MMoE on cached DINOv3 features for multi-task fish attributes")
    p.add_argument("--features-dir", type=Path, required=True, help="Directory with *_features.pt and metadata.json")
    p.add_argument("--ann-root", type=Path, required=True, help="Directory with train.csv and test.csv")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to store checkpoints and logs")

    # Optimization
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["none", "cosine", "step", "plateau", "linear"],
        default="none",
        help="Learning rate scheduler type",
    )
    p.add_argument("--lr-warmup-epochs", type=int, default=0, help="Warmup epochs for cosine scheduler")
    p.add_argument("--lr-step-size", type=int, default=10, help="Step size for StepLR")
    p.add_argument("--lr-gamma", type=float, default=0.1, help="Gamma for StepLR")

    # Model
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--expert-hidden", type=int, default=2048)
    p.add_argument("--tower-hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gating-temperature", type=float, default=1.0)

    # Loss weighting: [family, order, habitat, troph] (+ optional feedingpath)
    p.add_argument(
        "--loss-weights", type=float, nargs="+", default=[1.0, 0.5, 0.5, 0.5],
        help="Weights for [family, order, habitat, troph] (append one more if include-feedingpath)",
    )
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--class-weighting",
        choices=("none", "balanced"),
        default="none",
        help="Apply class-balanced weighting for multiclass heads; also sets habitat pos_weight",
    )

    # Data / preprocessing
    p.add_argument("--standardize", action="store_true", help="Z-score features using train stats")
    p.add_argument("--include-feedingpath", action="store_true", help="Add categorical FeedingPath head")

    # Misc
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="fishnet")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--exp-name", type=str, default=None)

    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA requested but not available")
    return args


# -----------------------
# Utilities
# -----------------------

def load_features(path: Path) -> Tuple[torch.Tensor, List[str]]:
    pack = torch.load(path, map_location="cpu")
    feats = pack["features"].float()
    paths = list(pack["paths"])  # list[str]
    return feats, paths


def standardize_features(train: torch.Tensor, *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    out = [(train - mean) / std]
    for x in others:
        out.append((x - mean) / std)
    return tuple(out)


def make_loader(features: torch.Tensor, labels: Dict[str, torch.Tensor], batch_size: int) -> DataLoader:
    # Build a tensor dataset with features + all label/mask tensors
    tensors = [features]
    keys = sorted(labels.keys())
    for k in keys:
        tensors.append(labels[k])
    ds = TensorDataset(*tensors)

    # Save ordering in the dataset object (hacky but handy)
    ds.label_keys = keys  # type: ignore[attr-defined]
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def unpack_batch(batch, keys: List[str]):
    features = batch[0]
    lbl_tensors = batch[1:]
    labels = {k: t for k, t in zip(keys, lbl_tensors)}
    return features, labels


def compute_multiclass_weights(labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropy.

    labels: LongTensor [N] with class indices (>=0)
    mask: FloatTensor [N] where >0 denotes valid
    """
    if mask is not None:
        idx = (mask > 0) & (labels >= 0)
        lbl = labels[idx]
    else:
        lbl = labels[labels >= 0]
    if lbl.numel() == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    num_classes = int(lbl.max().item()) + 1
    counts = torch.bincount(lbl, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * max(1, num_classes))
    return weights


def compute_habitat_pos_weight(hab: torch.Tensor) -> torch.Tensor:
    """Compute BCEWithLogits pos_weight as (N-P)/P per label.

    hab: FloatTensor [N, C] with 0/1 labels
    """
    N = max(1, hab.shape[0])
    pos = hab.sum(dim=0).float()
    neg = torch.tensor(float(N)).to(pos) - pos
    pw = neg / pos.clamp_min(1.0)
    return pw


@torch.no_grad()
def compute_metrics(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # Family accuracy
    if "family" in outputs and "family" in labels and "mask_family" in labels:
        mask = labels["mask_family"] > 0
        if mask.any():
            preds = outputs["family"].argmax(dim=1)
            acc = (preds[mask] == labels["family"][mask]).float().mean().item()
            metrics["acc_family"] = acc

    # Order accuracy
    if "order" in outputs and "order" in labels and "mask_order" in labels:
        mask = labels["mask_order"] > 0
        if mask.any():
            preds = outputs["order"].argmax(dim=1)
            acc = (preds[mask] == labels["order"][mask]).float().mean().item()
            metrics["acc_order"] = acc

    # Habitat F1@0.5 (macro over 8 labels)
    if "habitat" in outputs and "habitat" in labels and "mask_habitat" in labels:
        logits = outputs["habitat"]
        target = labels["habitat"].float()
        pred = (torch.sigmoid(logits) >= 0.5).float()
        tp = (pred * target).sum(dim=0)
        fp = (pred * (1 - target)).sum(dim=0)
        fn = ((1 - pred) * target).sum(dim=0)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        metrics["f1_habitat_macro"] = f1.mean().item()

    # Troph MAE
    if "troph" in outputs and "troph" in labels and "mask_troph" in labels:
        mask = labels["mask_troph"] > 0
        if mask.any():
            mae = (outputs["troph"].squeeze(-1)[mask] - labels["troph"][mask]).abs().mean().item()
            metrics["mae_troph"] = mae

    # FeedingPath accuracy (optional)
    if "feedingpath" in outputs and "feedingpath" in labels and "mask_feedingpath" in labels:
        mask = labels["mask_feedingpath"] > 0
        if mask.any():
            preds = outputs["feedingpath"].argmax(dim=1)
            acc = (preds[mask] == labels["feedingpath"][mask]).float().mean().item()
            metrics["acc_feedingpath"] = acc

    return metrics


# -----------------------
# Training and evaluation
# -----------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, criterions: Dict[str, nn.Module], loss_weights: Dict[str, float], device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        features, labels = unpack_batch(batch, loader.dataset.label_keys)  # type: ignore[attr-defined]
        features = features.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        outs = model(features)
        loss = 0.0

        # Family
        if "family" in outs:
            mask = labels.get("mask_family")
            if mask is not None and mask.sum() > 0:
                loss_fam = criterions["family"](outs["family"], labels["family"]) * mask
                loss += loss_weights["family"] * (loss_fam.sum() / mask.sum())

        # Order
        if "order" in outs:
            mask = labels.get("mask_order")
            if mask is not None and mask.sum() > 0:
                loss_ord = criterions["order"](outs["order"], labels["order"]) * mask
                loss += loss_weights["order"] * (loss_ord.sum() / mask.sum())

        # Habitat
        if "habitat" in outs:
            mask = labels.get("mask_habitat")
            logits = outs["habitat"]
            target = labels["habitat"].float()
            loss_hab = criterions["habitat"](logits, target)
            if mask is not None:
                loss_hab = (loss_hab.mean(dim=1) * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss_hab = loss_hab.mean()
            loss += loss_weights["habitat"] * loss_hab

        # Troph
        if "troph" in outs:
            mask = labels.get("mask_troph")
            pred = outs["troph"].squeeze(-1)
            target = labels["troph"]
            if mask is not None and mask.sum() > 0:
                loss_tr = criterions["troph"](pred[mask > 0], target[mask > 0])
                loss += loss_weights["troph"] * loss_tr

        # FeedingPath (optional)
        if "feedingpath" in outs and "feedingpath" in criterions:
            mask = labels.get("mask_feedingpath")
            if mask is not None and mask.sum() > 0:
                loss_fp = criterions["feedingpath"](outs["feedingpath"], labels["feedingpath"]) * mask
                loss += loss_weights.get("feedingpath", 0.0) * (loss_fp.sum() / mask.sum())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += float(loss.item())
        total_batches += 1

    return {"loss": total_loss / max(1, total_batches)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterions: Dict[str, nn.Module], loss_weights: Dict[str, float], device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    agg_metrics: Dict[str, float] = {}
    agg_counts: Dict[str, int] = {}

    for batch in loader:
        features, labels = unpack_batch(batch, loader.dataset.label_keys)  # type: ignore[attr-defined]
        features = features.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        outs = model(features)

        # Loss (for logging)
        loss = 0.0
        # Family
        if "family" in outs:
            mask = labels.get("mask_family")
            if mask is not None and mask.sum() > 0:
                loss_fam = criterions["family"](outs["family"], labels["family"]) * mask
                loss += loss_weights["family"] * (loss_fam.sum() / mask.sum())
        # Order
        if "order" in outs:
            mask = labels.get("mask_order")
            if mask is not None and mask.sum() > 0:
                loss_ord = criterions["order"](outs["order"], labels["order"]) * mask
                loss += loss_weights["order"] * (loss_ord.sum() / mask.sum())
        # Habitat
        if "habitat" in outs:
            mask = labels.get("mask_habitat")
            logits = outs["habitat"]
            target = labels["habitat"].float()
            loss_hab = criterions["habitat"](logits, target)
            if mask is not None:
                loss_hab = (loss_hab.mean(dim=1) * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss_hab = loss_hab.mean()
            loss += loss_weights["habitat"] * loss_hab
        # Troph
        if "troph" in outs:
            mask = labels.get("mask_troph")
            pred = outs["troph"].squeeze(-1)
            target = labels["troph"]
            if mask is not None and mask.sum() > 0:
                loss_tr = criterions["troph"](pred[mask > 0], target[mask > 0])
                loss += loss_weights["troph"] * loss_tr
        # FeedingPath
        if "feedingpath" in outs and "feedingpath" in criterions:
            mask = labels.get("mask_feedingpath")
            if mask is not None and mask.sum() > 0:
                loss_fp = criterions["feedingpath"](outs["feedingpath"], labels["feedingpath"]) * mask
                loss += loss_weights.get("feedingpath", 0.0) * (loss_fp.sum() / mask.sum())

        total_loss += float(loss)
        total_batches += 1

        # Metrics
        m = compute_metrics(outs, labels)
        for k, v in m.items():
            agg_metrics[k] = agg_metrics.get(k, 0.0) + float(v)
            agg_counts[k] = agg_counts.get(k, 0) + 1

    metrics = {k: agg_metrics[k] / max(1, agg_counts[k]) for k in agg_metrics}
    metrics["loss"] = total_loss / max(1, total_batches)
    return metrics


# -----------------------
# Main
# -----------------------

def main() -> None:
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

    args = parse_args()
    torch.manual_seed(args.seed)

    # Prepare experiment directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"mmoe_e{args.experts}_lr{args.lr:.0e}_bs{args.batch_size}_{timestamp}".replace("e-0", "e-")
    exp_dir = args.output_dir / args.exp_name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = vars(args).copy()
    for k, v in list(cfg.items()):
        if isinstance(v, Path):
            cfg[k] = str(v)
    (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Wandb
    if args.wandb and not WANDB_AVAILABLE:
        logger.warning("wandb requested but not installed; disabling")
        args.wandb = False
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name, dir=str(exp_dir), config=cfg)

    # Paths
    features_dir: Path = args.features_dir
    train_pt = features_dir / "train_features.pt"
    val_pt = features_dir / "val_features.pt"
    test_pt = features_dir / "test_features.pt"
    metadata_json = features_dir / "metadata.json"

    # Label encoders
    labelers = build_label_encoders(args.ann_root / "train.csv", args.ann_root / "test.csv", metadata_json=metadata_json, include_feedingpath=args.include_feedingpath)

    # Load features
    train_features, _ = load_features(train_pt)
    val_features, _ = load_features(val_pt) if val_pt.exists() else (train_features.clone(), [])
    test_features, _ = load_features(test_pt) if test_pt.exists() else (None, [])

    # Build labels per split (aligning via 'paths')
    train_labels = build_split_labels(train_pt, args.ann_root / "train.csv", labelers, include_feedingpath=args.include_feedingpath)
    val_labels = build_split_labels(val_pt, args.ann_root / "train.csv", labelers, include_feedingpath=args.include_feedingpath) if val_pt.exists() else train_labels
    test_labels = build_split_labels(test_pt, args.ann_root / "test.csv", labelers, include_feedingpath=args.include_feedingpath) if test_pt.exists() else None

    # Standardize
    if args.standardize:
        packs = [val_features]
        if test_features is not None:
            packs.append(test_features)
        std_feats = standardize_features(train_features, *packs)
        train_features = std_feats[0]
        val_features = std_feats[1]
        if test_features is not None:
            test_features = std_feats[2]

    # Dataloaders
    train_loader = make_loader(train_features, train_labels, args.batch_size)
    val_loader = make_loader(val_features, val_labels, args.batch_size)
    test_loader = make_loader(test_features, test_labels, args.batch_size) if test_features is not None and test_labels is not None else None

    # Model
    in_dim = train_features.size(1)
    tasks = [
        TaskSpec(name="family", type="multiclass", out_dim=len(labelers.family_to_idx)),
        TaskSpec(name="order", type="multiclass", out_dim=len(labelers.order_to_idx)),
        TaskSpec(name="habitat", type="multilabel", out_dim=len(HABITAT_BINARY_COLS)),
        TaskSpec(name="troph", type="regression", out_dim=1),
    ]
    if args.include_feedingpath and labelers.feeding_to_idx is not None:
        tasks.append(TaskSpec(name="feedingpath", type="multiclass", out_dim=len(labelers.feeding_to_idx)))

    model = GatedMMoE(
        in_dim=in_dim,
        experts=args.experts,
        expert_hidden=args.expert_hidden,
        tower_hidden=args.tower_hidden,
        tasks=tasks,
        dropout=args.dropout,
        gating_temperature=args.gating_temperature,
    ).to(args.device)

    # Losses (optionally class-balanced)
    family_weight = None
    order_weight = None
    feeding_weight = None
    habitat_pos_weight = None

    if args.class_weighting == "balanced":
        try:
            family_weight = compute_multiclass_weights(train_labels["family"], train_labels.get("mask_family"))
            order_weight = compute_multiclass_weights(train_labels["order"], train_labels.get("mask_order"))
            if args.include_feedingpath and "feedingpath" in train_labels:
                feeding_weight = compute_multiclass_weights(train_labels["feedingpath"], train_labels.get("mask_feedingpath"))
            habitat_pos_weight = compute_habitat_pos_weight(train_labels["habitat"])  # [C]
            logger.info(
                "Using class-balanced weighting. family_weight len=%s, order_weight len=%s, habitat_pos_weight shape=%s",
                None if family_weight is None else len(family_weight),
                None if order_weight is None else len(order_weight),
                None if habitat_pos_weight is None else tuple(habitat_pos_weight.shape),
            )
        except Exception as e:
            logger.warning("Failed to compute class-balanced weights: %s", e)
            family_weight = order_weight = feeding_weight = habitat_pos_weight = None

    # Ensure weights are on the right device
    dev = torch.device(args.device)
    if family_weight is not None:
        family_weight = family_weight.to(dev)
    if order_weight is not None:
        order_weight = order_weight.to(dev)
    if feeding_weight is not None:
        feeding_weight = feeding_weight.to(dev)
    if habitat_pos_weight is not None:
        habitat_pos_weight = habitat_pos_weight.to(dev)

    criterions: Dict[str, nn.Module] = {
        "family": nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing, weight=family_weight),
        "order": nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing, weight=order_weight),
        "habitat": nn.BCEWithLogitsLoss(reduction="none", pos_weight=habitat_pos_weight),
        "troph": nn.SmoothL1Loss(reduction="mean"),
    }
    if args.include_feedingpath and any(t.name == "feedingpath" for t in tasks):
        criterions["feedingpath"] = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing, weight=feeding_weight)

    # Loss weights mapping
    lw = args.loss_weights
    expect = 4 + (1 if (args.include_feedingpath and any(t.name == "feedingpath" for t in tasks)) else 0)
    if len(lw) != expect:
        logger.warning("--loss-weights length (%d) != expected (%d); padding/truncating.", len(lw), expect)
        if len(lw) < expect:
            lw = lw + [0.5] * (expect - len(lw))
        lw = lw[:expect]
    loss_weights = {"family": lw[0], "order": lw[1], "habitat": lw[2], "troph": lw[3]}
    if expect == 5:
        loss_weights["feedingpath"] = lw[4]

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler
    scheduler = None
    if args.lr_scheduler != "none":
        if args.lr_scheduler == "cosine":
            if args.lr_warmup_epochs > 0:
                main = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs - args.lr_warmup_epochs), eta_min=1e-6)
                warm = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.01, end_factor=1.0, total_iters=args.lr_warmup_epochs)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warm, main], milestones=[args.lr_warmup_epochs])
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
        elif args.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.01, total_iters=args.epochs)
        elif args.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5, verbose=True)
        else:
            raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")

    best_val_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optim, criterions, loss_weights, torch.device(args.device))
        val_stats = evaluate(model, val_loader, criterions, loss_weights, torch.device(args.device))

        # Composite validation score to early-stop: family acc + 0.5*order acc + 0.5*f1_habitat - 0.1*mae
        fam = val_stats.get("acc_family", 0.0)
        ordacc = val_stats.get("acc_order", 0.0)
        f1h = val_stats.get("f1_habitat_macro", 0.0)
        mae = val_stats.get("mae_troph", 0.0)
        comp = fam + 0.5 * ordacc + 0.5 * f1h - 0.1 * mae

        # Step the scheduler
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(comp)
            else:
                scheduler.step()
        current_lr = optim.param_groups[0]['lr']

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | fam=%.3f | ord=%.3f | f1hab=%.3f | mae=%.3f | comp=%.3f | lr=%.6f",
            epoch, train_stats["loss"], val_stats.get("loss", 0.0), fam, ordacc, f1h, mae, comp, current_lr,
        )

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "val/loss": val_stats.get("loss", 0.0),
                "val/acc_family": fam,
                "val/acc_order": ordacc,
                "val/f1_habitat_macro": f1h,
                "val/mae_troph": mae,
                "val/composite": comp,
                "lr": current_lr,
            })

        if comp > best_val_score + 1e-6:
            best_val_score = comp
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            # Save checkpoint
            ckpt = exp_dir / "checkpoints" / "best_model.pt"
            torch.save(best_state, ckpt)
        else:
            patience += 1
            if patience >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best and final evaluation
    model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, criterions, loss_weights, torch.device(args.device))
    results = {
        "val": final_val,
        "config": cfg,
    }

    if test_loader is not None:
        test_stats = evaluate(model, test_loader, criterions, loss_weights, torch.device(args.device))
        results["test"] = test_stats

    # Save results
    (exp_dir / "results" / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", exp_dir / "results" / "metrics.json")

    if args.wandb:
        for k, v in results.get("val", {}).items():
            wandb.summary[f"best/{k}"] = v
        if "test" in results:
            for k, v in results["test"].items():
                wandb.summary[f"test/{k}"] = v
        wandb.finish()


if __name__ == "__main__":
    main()
