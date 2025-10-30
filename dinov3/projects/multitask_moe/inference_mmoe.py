"""
Inference script for MMoE multi-task fish attribute prediction.
Visualizes predictions on random sample of images.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .models import GatedMMoE, TaskSpec
from .data import HABITAT_BINARY_COLS, build_label_encoders


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[GatedMMoE, Dict]:
    """Load model checkpoint and metadata."""
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Load config from same directory
    config_path = checkpoint_path.parent.parent / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    # Get input dimension from state dict
    in_dim = state_dict["experts.0.1.weight"].shape[1]
    
    # Build task specs from state dict
    tasks = []
    
    # Check what tasks exist in the model
    task_names = set()
    for key in state_dict.keys():
        if key.startswith("task_towers."):
            task_name = key.split(".")[1]
            task_names.add(task_name)
    
    # Family - look for the final output layer
    if "family" in task_names:
        # The final layer is task_towers.family.3.2.weight
        family_dim = state_dict["task_towers.family.3.2.weight"].shape[0]
        tasks.append(TaskSpec(name="family", type="multiclass", out_dim=family_dim))
    
    # Order
    if "order" in task_names:
        order_dim = state_dict["task_towers.order.3.2.weight"].shape[0]
        tasks.append(TaskSpec(name="order", type="multiclass", out_dim=order_dim))
    
    # Habitat
    if "habitat" in task_names:
        habitat_dim = state_dict["task_towers.habitat.3.2.weight"].shape[0]
        tasks.append(TaskSpec(name="habitat", type="multilabel", out_dim=habitat_dim))
    
    # Troph
    if "troph" in task_names:
        tasks.append(TaskSpec(name="troph", type="regression", out_dim=1))
    
    # FeedingPath
    if "feedingpath" in task_names:
        feeding_dim = state_dict["task_towers.feedingpath.3.2.weight"].shape[0]
        tasks.append(TaskSpec(name="feedingpath", type="multiclass", out_dim=feeding_dim))
    
    # Create model
    model = GatedMMoE(
        in_dim=in_dim,
        experts=config.get("experts", 4),
        expert_hidden=config.get("expert_hidden", 2048),
        tower_hidden=config.get("tower_hidden", 1024),
        tasks=tasks,
        dropout=config.get("dropout", 0.1),
        gating_temperature=config.get("gating_temperature", 1.0),
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config


def load_features_and_paths(features_pt: Path) -> Tuple[torch.Tensor, List[str]]:
    """Load features and image paths."""
    pack = torch.load(features_pt, map_location="cpu")
    features = pack["features"].float()
    paths = list(pack["paths"])
    return features, paths


def standardize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply standardization using pre-computed statistics."""
    return (features - mean) / std


def get_train_statistics(train_features_pt: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load training set statistics for standardization."""
    features, _ = load_features_and_paths(train_features_pt)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return mean, std


def sample_images_per_class(
    csv_path: Path,
    dataset_root: Path,
    n_classes: int = 50,
    images_per_class: int = 1,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Sample random images from random classes.
    
    Returns:
        List of (family_name, image_name, full_path) tuples
    """
    random.seed(seed)
    
    # Read CSV and group by family
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Group by family
    family_to_images = {}
    for row in rows:
        family = row.get("Folder", "").strip()
        image = row.get("image", "").strip()
        if not family or not image:
            continue
        
        # Normalize image name
        if image.startswith("http"):
            image = image.split("/")[-1]
        
        # Check if image exists
        img_path = dataset_root / family / image
        if img_path.exists():
            if family not in family_to_images:
                family_to_images[family] = []
            family_to_images[family].append((image, str(img_path), row))
    
    # Sample random classes
    available_families = [f for f in family_to_images.keys() if len(family_to_images[f]) >= images_per_class]
    if len(available_families) < n_classes:
        print(f"Warning: Only {len(available_families)} families have {images_per_class}+ images")
        n_classes = len(available_families)
    
    selected_families = random.sample(available_families, n_classes)
    
    # Sample images from each family
    samples = []
    for family in selected_families:
        family_images = family_to_images[family]
        selected_images = random.sample(family_images, min(images_per_class, len(family_images)))
        for image_name, full_path, row in selected_images:
            samples.append((family, image_name, full_path, row))
    
    return samples


def get_ground_truth(row: Dict[str, str]) -> Dict[str, str]:
    """Extract ground truth labels from CSV row."""
    gt = {
        "family": row.get("Folder", "Unknown"),
        "order": row.get("Order", "Unknown"),
        "troph": row.get("Troph", "Unknown"),
        "feedingpath": row.get("FeedingPath", "Unknown"),
    }
    
    # Habitat (multi-label)
    habitat_parts = []
    for col in HABITAT_BINARY_COLS:
        if row.get(col, "0") == "1":
            habitat_parts.append(col)
    gt["habitat"] = ", ".join(habitat_parts) if habitat_parts else "Unknown"
    
    return gt


@torch.no_grad()
def predict_on_features(
    model: GatedMMoE,
    features: torch.Tensor,
    idx_to_family: Dict[int, str],
    idx_to_order: Dict[int, str],
    idx_to_feeding: Dict[int, str] | None = None,
) -> Dict[str, str]:
    """Run inference and decode predictions."""
    features = features.unsqueeze(0)  # [1, D]
    outputs = model(features)
    
    predictions = {}
    
    # Family
    if "family" in outputs:
        family_logits = outputs["family"][0]
        family_probs = F.softmax(family_logits, dim=0)
        family_idx = family_probs.argmax().item()
        family_conf = family_probs.max().item()
        predictions["family"] = f"{idx_to_family.get(family_idx, 'Unknown')} ({family_conf:.2f})"
    
    # Order
    if "order" in outputs:
        order_logits = outputs["order"][0]
        order_probs = F.softmax(order_logits, dim=0)
        order_idx = order_probs.argmax().item()
        order_conf = order_probs.max().item()
        predictions["order"] = f"{idx_to_order.get(order_idx, 'Unknown')} ({order_conf:.2f})"
    
    # Habitat (multi-label)
    if "habitat" in outputs:
        habitat_logits = outputs["habitat"][0]
        habitat_probs = torch.sigmoid(habitat_logits)
        habitat_preds = (habitat_probs >= 0.5).cpu().numpy()
        habitat_labels = [HABITAT_BINARY_COLS[i] for i, v in enumerate(habitat_preds) if v]
        if habitat_labels:
            # Show top 3 with confidence
            top_indices = habitat_probs.topk(min(3, len(habitat_probs))).indices
            habitat_parts = [f"{HABITAT_BINARY_COLS[i]} ({habitat_probs[i]:.2f})" for i in top_indices if habitat_probs[i] >= 0.5]
            predictions["habitat"] = ", ".join(habitat_parts) if habitat_parts else "None"
        else:
            predictions["habitat"] = "None"
    
    # Troph (regression)
    if "troph" in outputs:
        troph_val = outputs["troph"][0, 0].item()
        predictions["troph"] = f"{troph_val:.2f}"
    
    # FeedingPath
    if "feedingpath" in outputs and idx_to_feeding:
        feeding_logits = outputs["feedingpath"][0]
        feeding_probs = F.softmax(feeding_logits, dim=0)
        feeding_idx = feeding_probs.argmax().item()
        feeding_conf = feeding_probs.max().item()
        predictions["feedingpath"] = f"{idx_to_feeding.get(feeding_idx, 'Unknown')} ({feeding_conf:.2f})"
    
    return predictions


def draw_predictions_on_image(
    image_path: str,
    predictions: Dict[str, str],
    ground_truth: Dict[str, str],
    output_path: Path,
    max_width: int = 1200,
) -> None:
    """Draw predictions and ground truth on image and save."""
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Resize if too large
    if img.width > max_width:
        scale = max_width / img.width
        new_size = (max_width, int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create drawing context
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
    
    # Prepare text
    text_lines = ["PREDICTIONS:"]
    for key in ["family", "order", "habitat", "troph"]:
        if key in predictions:
            text_lines.append(f"{key.upper()}: {predictions[key]}")
    
    text_lines.append("")
    text_lines.append("GROUND TRUTH:")
    for key in ["family", "order", "habitat", "troph"]:
        if key in ground_truth:
            text_lines.append(f"{key.upper()}: {ground_truth[key]}")
    
    # Calculate text dimensions
    line_height = 24
    margin = 15
    text_height = len(text_lines) * line_height + 2 * margin
    
    # Create new image with space for text
    new_height = img.height + text_height
    new_img = Image.new("RGB", (img.width, new_height), color=(255, 255, 255))
    new_img.paste(img, (0, 0))
    
    # Draw text on white background
    draw = ImageDraw.Draw(new_img)
    y_offset = img.height + margin
    
    for i, line in enumerate(text_lines):
        if line in ["PREDICTIONS:", "GROUND TRUTH:"]:
            # Title lines in bold
            draw.text((margin, y_offset + i * line_height), line, fill=(0, 0, 0), font=font_title)
        elif line == "":
            continue
        else:
            draw.text((margin, y_offset + i * line_height), line, fill=(50, 50, 50), font=font_text)
    
    # Save
    new_img.save(output_path, quality=95)
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MMoE inference and visualization")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--features-dir", type=Path, required=True, help="Directory with features")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory with images")
    parser.add_argument("--ann-csv", type=Path, required=True, help="CSV with annotations (train or test)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mmoe_visualizations"), help="Output directory")
    parser.add_argument("--n-classes", type=int, default=50, help="Number of random classes to sample")
    parser.add_argument("--images-per-class", type=int, default=1, help="Images per class")
    parser.add_argument("--standardize", action="store_true", help="Standardize features using train stats")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Load label encoders
    print("Building label encoders")
    ann_root = args.ann_csv.parent
    metadata_json = args.features_dir / "metadata.json"
    labelers = build_label_encoders(
        ann_root / "train.csv",
        ann_root / "test.csv",
        metadata_json=metadata_json if metadata_json.exists() else None,
        include_feedingpath=False,
    )
    
    # Create reverse mappings
    idx_to_family = {v: k for k, v in labelers.family_to_idx.items()}
    idx_to_order = {v: k for k, v in labelers.order_to_idx.items()}
    idx_to_feeding = None
    if labelers.feeding_to_idx:
        idx_to_feeding = {v: k for k, v in labelers.feeding_to_idx.items()}
    
    # Load features
    print("Loading features")
    split_name = args.ann_csv.stem  # train or test
    features_pt = args.features_dir / f"{split_name}_features.pt"
    features, paths = load_features_and_paths(features_pt)
    
    # Standardize if needed
    if args.standardize:
        print("Standardizing features")
        train_features_pt = args.features_dir / "train_features.pt"
        mean, std = get_train_statistics(train_features_pt)
        features = standardize_features(features, mean, std)
    
    # Build path to feature index mapping
    path_to_idx = {p: i for i, p in enumerate(paths)}
    
    # Sample images
    print(f"Sampling {args.n_classes} classes with {args.images_per_class} image(s) each")
    samples = sample_images_per_class(
        args.ann_csv,
        args.dataset_root,
        n_classes=args.n_classes,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )
    
    print(f"Found {len(samples)} images to process")
    
    # Process each image
    for i, (family, image_name, full_path, row) in enumerate(samples):
        print(f"\nProcessing {i+1}/{len(samples)}: {family}/{image_name}")
        
        # Get ground truth
        gt = get_ground_truth(row)
        
        # Find corresponding features
        # Features are stored with relative paths like "Family/image.jpg"
        rel_path = f"{family}/{image_name}"
        
        # Try to find matching feature
        feat_idx = None
        for path_key, idx in path_to_idx.items():
            if path_key.endswith(rel_path) or path_key.endswith(image_name):
                feat_idx = idx
                break
        
        if feat_idx is None:
            print(f"  Warning: Could not find features for {rel_path}")
            continue
        
        # Get features
        feat = features[feat_idx].to(device)
        
        # Run inference
        predictions = predict_on_features(
            model, feat, idx_to_family, idx_to_order, idx_to_feeding
        )
        
        # Print predictions
        print(f"  Predictions: {predictions}")
        print(f"  Ground Truth: {gt}")
        
        # Draw and save
        output_filename = f"{family}_{image_name}"
        output_path = args.output_dir / output_filename
        
        try:
            draw_predictions_on_image(full_path, predictions, gt, output_path)
        except Exception as e:
            print(f"  Error drawing image: {e}")
            continue
    
    print(f"\n✓ Done! Saved {len(samples)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
