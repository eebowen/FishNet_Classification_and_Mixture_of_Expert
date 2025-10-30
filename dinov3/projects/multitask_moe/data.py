from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

HABITAT_BINARY_COLS = [
    "Tropical",
    "Temperate",
    "Subtropical",
    "Boreal",
    "Polar",
    "freshwater",
    "saltwater",
    "brackish",
]

@dataclass
class LabelEncoders:
    family_to_idx: Dict[str, int]
    order_to_idx: Dict[str, int]
    feeding_to_idx: Optional[Dict[str, int]] = None


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _normalize_image_name(s: str) -> str:
    if s.startswith("http"):
        s = s.split("/")[-1]
    return s


def build_label_encoders(train_csv: Path, test_csv: Path, metadata_json: Optional[Path] = None, include_feedingpath: bool = False) -> LabelEncoders:
    rows = _read_csv_rows(train_csv) + _read_csv_rows(test_csv)
    families = sorted({r["Folder"] for r in rows if r.get("Folder")})
    orders = sorted({r["Order"] for r in rows if r.get("Order")})

    # If feature metadata provides a class_to_idx for families, prefer that for stability
    family_to_idx: Dict[str, int] = {f: i for i, f in enumerate(families)}
    if metadata_json is not None and metadata_json.exists():
        meta = json.loads(metadata_json.read_text())
        if "class_to_idx" in meta and isinstance(meta["class_to_idx"], dict):
            # Ensure we keep only classes we know, preserving provided mapping order
            provided = meta["class_to_idx"]
            family_to_idx = {k: int(v) for k, v in provided.items() if k in families}

    order_to_idx: Dict[str, int] = {o: i for i, o in enumerate(orders)}

    feeding_to_idx = None
    if include_feedingpath:
        feeding_vals = sorted({(r.get("FeedingPath") or "").strip() for r in rows if r.get("FeedingPath")})
        feeding_vals = [v for v in feeding_vals if v]
        if feeding_vals:
            feeding_to_idx = {v: i for i, v in enumerate(feeding_vals)}

    return LabelEncoders(family_to_idx=family_to_idx, order_to_idx=order_to_idx, feeding_to_idx=feeding_to_idx)


def build_split_labels(
    features_pt: Path,
    ann_csv: Path,
    labelers: LabelEncoders,
    include_feedingpath: bool = False,
) -> Dict[str, torch.Tensor]:
    payload = torch.load(features_pt, map_location="cpu")
    paths: List[str] = payload["paths"]

    # Index CSV by (folder, image_name)
    by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in _read_csv_rows(ann_csv):
        folder = row.get("Folder", "").strip()
        image = _normalize_image_name((row.get("image") or "").strip())
        if folder and image:
            by_key[(folder, image)] = row

    family_labels: List[int] = []
    order_labels: List[int] = []
    habitat_labels: List[List[int]] = []
    troph_values: List[float] = []
    feeding_labels: List[int] = []

    missing = 0
    for p in paths:
        pth = Path(p)
        folder = pth.parent.name
        image = pth.name
        row = by_key.get((folder, image))
        if row is None:
            missing += 1
            # Fill with NaNs / -1
            family_labels.append(labelers.family_to_idx.get(folder, -1))
            order_labels.append(-1)
            habitat_labels.append([0] * len(HABITAT_BINARY_COLS))
            troph_values.append(float("nan"))
            if include_feedingpath and labelers.feeding_to_idx is not None:
                feeding_labels.append(-1)
            continue

        family_labels.append(labelers.family_to_idx.get(folder, -1))
        order_idx = labelers.order_to_idx.get(row.get("Order", ""), -1)
        order_labels.append(order_idx)

        # Habitat multi-label
        hl: List[int] = []
        for col in HABITAT_BINARY_COLS:
            val = row.get(col)
            if val is None or val == "":
                hl.append(0)
            else:
                try:
                    # Accept 0/1 or string '0'/'1'
                    hl.append(1 if int(float(val)) > 0 else 0)
                except Exception:
                    # Fallback: treat non-empty as positive
                    hl.append(1)
        habitat_labels.append(hl)

        # Troph
        tv = row.get("Troph")
        try:
            troph_values.append(float(tv) if tv not in (None, "") else float("nan"))
        except Exception:
            troph_values.append(float("nan"))

        if include_feedingpath and labelers.feeding_to_idx is not None:
            fp = (row.get("FeedingPath") or "").strip()
            feeding_labels.append(labelers.feeding_to_idx.get(fp, -1))

    out: Dict[str, torch.Tensor] = {
        "family": torch.tensor(family_labels, dtype=torch.long),
        "order": torch.tensor(order_labels, dtype=torch.long),
        "habitat": torch.tensor(habitat_labels, dtype=torch.float32),
        "troph": torch.tensor(troph_values, dtype=torch.float32),
    }
    if include_feedingpath and labelers.feeding_to_idx is not None:
        out["feedingpath"] = torch.tensor(feeding_labels, dtype=torch.long)

    # Masks (1 where label is available)
    out["mask_family"] = (out["family"] >= 0).to(torch.float32)
    out["mask_order"] = (out["order"] >= 0).to(torch.float32)
    out["mask_habitat"] = torch.ones(out["habitat"].shape[0], dtype=torch.float32)
    out["mask_troph"] = (~torch.isnan(out["troph"])) .to(torch.float32)
    if "feedingpath" in out:
        out["mask_feedingpath"] = (out["feedingpath"] >= 0).to(torch.float32)

    return out
