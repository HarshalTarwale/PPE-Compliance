"""
Optimized 5-Class PPE Dataset Generator.

Filters the 25-class Construction Site Safety dataset down to the 5 core safety classes:
  0: Person
  1: Hardhat
  2: NO-Hardhat
  3: Safety Vest
  4: NO-Safety Vest

Removes 20 non-PPE background classes (vehicles, machinery, cones, gloves, etc.),
eliminating class imbalance and concentrating model gradients purely on PPE compliance.

Usage:
  python scripts/prepare_5class_dataset.py --source "C:/path/to/dataset" --output data/processed/ppe_5class
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("prepare_dataset")

# Target class names in standard order
TARGET_NAMES: List[str] = [
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
]


def resolve_split_path(raw_path: str, dataset_root: Path) -> Path:
    """Resolve split path from data.yaml, handling Roboflow leading ../ issues."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    # Relative to dataset_root
    clean_parts = [part for part in p.parts if part != ".."]
    candidate = dataset_root.joinpath(*clean_parts)
    if candidate.exists():
        return candidate
    candidate2 = dataset_root / raw_path
    if candidate2.exists():
        return candidate2
    return p


def prepare_5class_dataset(source_dir: str, output_dir: str) -> Path:
    """Process original YOLO dataset and build clean 5-class target dataset."""
    src_root = Path(source_dir).resolve()
    yaml_path = src_root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        src_yaml = yaml.safe_load(f)

    src_names: List[str] = src_yaml.get("names", [])
    if isinstance(src_names, dict):
        src_names = [src_names[k] for k in sorted(src_names.keys())]

    logger.info("Found %d classes in source dataset: %s", len(src_names), src_names)

    # Build mapping from source class_id -> target class_id (0..4)
    class_mapping: Dict[int, int] = {}
    for target_idx, target_name in enumerate(TARGET_NAMES):
        if target_name in src_names:
            orig_idx = src_names.index(target_name)
            class_mapping[orig_idx] = target_idx
            logger.info("Mapped: '%s' (src ID %d) -> target ID %d", target_name, orig_idx, target_idx)
        else:
            logger.warning("Target class '%s' not found in source dataset!", target_name)

    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    stats: Dict[str, Dict[str, int]] = {s: {"images": 0, "labels": 0, "instances": 0} for s in splits}

    for split in splits:
        # Determine image dir from yaml or convention
        raw_split_key = "val" if split == "valid" and "val" in src_yaml else split
        raw_split_path = src_yaml.get(raw_split_key, split)
        src_img_dir = resolve_split_path(raw_split_path, src_root)

        if not src_img_dir.is_dir():
            # Fallback
            for cand_name in [f"{split}/images", f"{split}", f"valid/images" if split == "valid" else split]:
                cand = src_root / cand_name
                if cand.is_dir():
                    src_img_dir = cand
                    break

        if not src_img_dir.is_dir():
            logger.warning("Split directory for '%s' not found at %s. Skipping.", split, src_img_dir)
            continue

        src_lbl_dir = src_img_dir.parent / "labels"
        if not src_lbl_dir.is_dir():
            src_lbl_dir = src_root / split / "labels"

        out_img_dir = out_root / split / "images"
        out_lbl_dir = out_root / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing split '%s': images from %s, labels from %s", split, src_img_dir, src_lbl_dir)

        img_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

        for img_path in img_files:
            lbl_name = img_path.stem + ".txt"
            src_lbl_path = src_lbl_dir / lbl_name

            filtered_lines: List[str] = []
            if src_lbl_path.is_file():
                with open(src_lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                orig_cls = int(parts[0])
                                if orig_cls in class_mapping:
                                    new_cls = class_mapping[orig_cls]
                                    filtered_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
                            except ValueError:
                                continue

            # Write filtered label
            out_lbl_path = out_lbl_dir / lbl_name
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.writelines(filtered_lines)

            # Link or copy image
            out_img_path = out_img_dir / img_path.name
            if not out_img_path.exists():
                try:
                    os.link(str(img_path), str(out_img_path))  # Hard link (instant, 0 disk space)
                except Exception:
                    shutil.copy2(str(img_path), str(out_img_path))

            stats[split]["images"] += 1
            if filtered_lines:
                stats[split]["labels"] += 1
                stats[split]["instances"] += len(filtered_lines)

    # Write new data.yaml
    new_yaml_data = {
        "path": str(out_root).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_NAMES),
        "names": TARGET_NAMES,
    }

    out_yaml_path = out_root / "data.yaml"
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(new_yaml_data, f, sort_keys=False)

    logger.info("Successfully generated 5-class PPE dataset at: %s", out_root)
    for s, st in stats.items():
        logger.info(
            "Split '%s': %d images, %d active labels, %d target PPE instances",
            s,
            st["images"],
            st["labels"],
            st["instances"],
        )

    return out_yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 5-class focused PPE dataset.")
    parser.add_argument(
        "--source",
        type=str,
        default=r"C:\Users\bhara\Downloads\Construction Site Safety.v30-raw-images_latestversion.yolov8",
        help="Path to source Construction Site Safety dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/ppe_5class",
        help="Output directory for 5-class dataset.",
    )
    args = parser.parse_args()

    prepare_5class_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
