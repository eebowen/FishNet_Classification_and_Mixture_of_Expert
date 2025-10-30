"""Utilities to extract DINOv3 features for the FishNet dataset."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    make_classification_eval_transform,
)
from dinov3.distributed import is_enabled, is_main_process
from dinov3.eval.utils import extract_features, wrap_model
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")


@dataclass(frozen=True)
class SplitConfig:
    name: str
    directory: Path | None
    proportion: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv3 features for FishNet")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the FishNet dataset (expects class subfolders)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where extracted features and metadata will be stored.",
    )
    parser.add_argument(
        "--hub-name",
        type=str,
        default="dinov3_vits16",
        help=(
            "Name of the DINOv3 backbone exposed via torch.hub (e.g. dinov3_vits16, dinov3_vitb16, "
            "dinov3_convnext_base)."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path or URL to the pretrained checkpoint to load with the specified hub backbone.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used during feature extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers per process.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=RESIZE_DEFAULT_SIZE,
        help="Resize shorter side to this value before center crop (matches DINOv3 eval preset).",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=CROP_DEFAULT_SIZE,
        help="Output crop size after resizing (matches DINOv3 eval preset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used when automatic train/val/test splits are created.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio to use when train/val/test folders are absent.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test ratio to use when train/val/test folders are absent.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Force-enable distributed feature extraction (default when CUDA is available).",
    )
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="Disable L2-normalisation of features (enabled by default).",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable multi-scale feature extraction (3-scale averaging).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 autocast for faster extraction when supported.",
    )

    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        parser.error("val_ratio and test_ratio must be >= 0 and sum to < 1 when creating splits")

    has_cuda = torch.cuda.is_available()
    if args.distributed and not has_cuda:
        parser.error("--distributed requires CUDA-capable PyTorch")

    if not args.distributed and has_cuda:
        logger.info("Enabling distributed mode by default because CUDA is available.")
        args.distributed = True

    if not has_cuda:
        logger.warning("CUDA is not available; extraction will run on CPU without distributed collectives.")

    return args


def make_split_configs(dataset_root: Path, split_names: Sequence[str], val_ratio: float, test_ratio: float) -> List[SplitConfig]:
    configs: List[SplitConfig] = []
    for name in split_names:
        directory = (dataset_root / name) if (dataset_root / name).is_dir() else None
        configs.append(SplitConfig(name=name, directory=directory, proportion=None))

    if any(cfg.directory for cfg in configs):
        return configs

    train_ratio = 1.0 - val_ratio - test_ratio
    split_props = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    return [SplitConfig(name=name, directory=None, proportion=split_props[name]) for name in split_names]


def _resolve_imagefolder(dataset: Dataset) -> ImageFolder:
    if isinstance(dataset, ImageFolder):
        return dataset
    if isinstance(dataset, Subset):
        return _resolve_imagefolder(dataset.dataset)
    raise TypeError(f"Unsupported dataset container: {type(dataset)!r}")


def _collect_paths(dataset: Dataset) -> List[str]:
    if isinstance(dataset, ImageFolder):
        return [path for path, _ in dataset.samples]
    if isinstance(dataset, Subset):
        base_paths = _collect_paths(dataset.dataset)
        return [base_paths[idx] for idx in dataset.indices]
    raise TypeError(f"Unsupported dataset container: {type(dataset)!r}")


def _collect_indices(dataset: Dataset) -> List[int]:
    if isinstance(dataset, ImageFolder):
        return list(range(len(dataset)))
    if isinstance(dataset, Subset):
        return [dataset.indices[i] for i in range(len(dataset))]
    raise TypeError(f"Unsupported dataset container: {type(dataset)!r}")


def build_datasets(
    dataset_root: Path,
    transform,
    split_configs: Sequence[SplitConfig],
    seed: int,
) -> Tuple[Mapping[str, Dataset], List[str], Dict[str, int]]:
    datasets: MutableMapping[str, Dataset] = {}
    manual_split = all(cfg.directory is not None for cfg in split_configs)

    if manual_split:
        for cfg in split_configs:
            assert cfg.directory is not None
            datasets[cfg.name] = ImageFolder(cfg.directory, transform=transform)

        all_classes = sorted({cls for ds in datasets.values() for cls in _resolve_imagefolder(ds).classes})
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

        for dataset in datasets.values():
            imagefolder = _resolve_imagefolder(dataset)
            mapping = {local_idx: class_to_idx[cls] for local_idx, cls in enumerate(imagefolder.classes)}

            def _target_transform(target, local_to_global=mapping):
                return local_to_global[int(target)]

            imagefolder.target_transform = _target_transform

        return datasets, all_classes, class_to_idx

    base_dataset = ImageFolder(dataset_root, transform=transform)
    split_lengths: List[int] = []
    for cfg in split_configs:
        if cfg.proportion is None:
            raise ValueError("Split proportions must be provided when explicit directories are missing")
        split_lengths.append(int(math.floor(cfg.proportion * len(base_dataset))))

    remainder = len(base_dataset) - sum(split_lengths)
    split_lengths[0] += remainder

    generator = torch.Generator().manual_seed(seed)
    subsets = random_split(base_dataset, split_lengths, generator=generator)

    for cfg, subset in zip(split_configs, subsets):
        datasets[cfg.name] = subset
    return datasets, list(base_dataset.classes), dict(base_dataset.class_to_idx)


def load_backbone(hub_name: str, weights: str | Path, normalize: bool, multi_scale: bool) -> torch.nn.Module:
    weights_arg = os.fspath(weights)
    script_path = Path(__file__).resolve()
    repo_root = next(
        (parent for parent in script_path.parents if (parent / "hubconf.py").exists()),
        None,
    )
    if repo_root is None:
        raise FileNotFoundError("Unable to locate hubconf.py relative to extract_features.py")

    model = torch.hub.load(repo_root.as_posix(), hub_name, source="local", weights=weights_arg)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    model = wrap_model(model, normalize=normalize, multi_scale=multi_scale)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def extract_split_features(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    use_fp16: bool,
) -> Dict[str, torch.Tensor]:
    if use_fp16 and not torch.cuda.is_available():
        logger.warning("FP16 requested but CUDA is unavailable; defaulting to float32.")
        use_fp16 = False

    if is_enabled():
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_fp16 else nullcontext()
        with autocast_ctx:
            features, labels = extract_features(
                model,
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                gather_on_cpu=True,
            )
        return {"features": features.cpu(), "labels": labels.cpu()}

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    feature_shards: List[torch.Tensor] = []
    label_shards: List[torch.Tensor] = []
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if (use_fp16 and torch.cuda.is_available()) else nullcontext()

    with torch.inference_mode():
        with autocast_ctx:
            for images, labels in data_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images).float()
                feature_shards.append(outputs.cpu())
                label_shards.append(labels.detach().cpu())

    features = torch.cat(feature_shards, dim=0)
    labels = torch.cat(label_shards, dim=0)
    return {"features": features, "labels": labels}


def main() -> None:
    args = parse_args()

    splits = ("train", "val", "test")
    split_configs = make_split_configs(args.dataset_root, splits, args.val_ratio, args.test_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with job_context(
        output_dir=os.fspath(args.output_dir),
        distributed_enabled=args.distributed,
        seed=args.seed,
    ):
        transform = make_classification_eval_transform(
            resize_size=args.resize_size,
            crop_size=args.crop_size,
        )
        datasets, class_names, class_to_idx = build_datasets(args.dataset_root, transform, split_configs, args.seed)
        backbone = load_backbone(
            hub_name=args.hub_name,
            weights=args.weights,
            normalize=not args.disable_normalize,
            multi_scale=args.multi_scale,
        )

        metadata: Dict[str, Dict[str, Iterable]] = {}
        for split_name, dataset in datasets.items():
            if len(dataset) == 0:
                logger.warning("Skipping split '%s' because it is empty", split_name)
                continue

            logger.info("Extracting features for split '%s' with %d samples", split_name, len(dataset))
            package = extract_split_features(
                model=backbone,
                dataset=dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_fp16=args.fp16,
            )

            if not is_main_process():
                continue

            split_paths = _collect_paths(dataset)
            split_indices = _collect_indices(dataset)
            features_path = args.output_dir / f"{split_name}_features.pt"
            torch.save(
                {
                    "features": package["features"],
                    "labels": package["labels"],
                    "paths": split_paths,
                    "indices": split_indices,
                },
                features_path,
            )
            logger.info("Saved %s", features_path)

            metadata[split_name] = {
                "count": len(split_paths),
                "features_file": features_path.name,
            }

        if is_main_process():
            metadata_output = {
                "dataset_root": os.fspath(args.dataset_root.resolve()),
                "class_to_idx": class_to_idx,
                "classes": class_names,
                "transform": {"resize_size": args.resize_size, "crop_size": args.crop_size},
                "splits": metadata,
                "hub_name": args.hub_name,
                "weights": os.fspath(args.weights),
            }
            metadata_path = args.output_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata_output, indent=2))
            logger.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
