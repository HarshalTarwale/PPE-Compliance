"""
Life Jacket Dataset Downloader & Merger.

Downloads a life-jacket detection dataset from Roboflow Universe or loads a
local YOLO-formatted dataset, and merges it into the PPE compliance dataset.

By default, operates in 5-Class Mode:
  In this mode, life jackets serve as torso protection (Safety Vests).
  The model is trained on life jacket images but classifies them under the
  existing "Safety Vest" / "NO-Safety Vest" classes:

    0: Person
    1: Hardhat
    2: NO-Hardhat
    3: Safety Vest     (detects Safety Vests and Life Jackets)
    4: NO-Safety Vest  (detects absence of Safety Vest / Life Jacket)

  This ensures 100% plug-and-play compatibility with the existing:
    - Frontend UI (already displays "Safety Vest" without any changes)
    - Backend API & WebSocket streams
    - Compliance rule engine (verifies torso protection)
    - Database schemas & alert notifications

Optionally, `--separate-classes` can be passed to generate a 7-class dataset
with separate "Life Jacket" and "NO-Life Jacket" categories.

Usage:
  # Merge from a local YOLO-format dataset directory (into 5-class dataset):
  python scripts/add_lifejacket_dataset.py --source "C:/path/to/lifejacket-dataset"

  # Merge directly in-place into existing data/processed/ppe_5class:
  python scripts/add_lifejacket_dataset.py --source "C:/path/to/lifejacket-dataset" --in-place

  # Download from Roboflow Universe and merge:
  python scripts/add_lifejacket_dataset.py \\
      --api-key YOUR_KEY --workspace WORKSPACE --project PROJECT --version 1

  # Specify custom class names from the source dataset:
  python scripts/add_lifejacket_dataset.py --source "C:/path/to/dataset" \\
      --positive-class "wear" --negative-class "notwear"
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("add_lifejacket")

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants & Target Class Definitions
# ---------------------------------------------------------------------------

TARGET_NAMES_5CLASS: List[str] = [
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
]

TARGET_NAMES_7CLASS: List[str] = [
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
    "Life Jacket",
    "NO-Life Jacket",
]

# Aliases for positive torso PPE (Life Jacket / Life Vest / Safety Vest)
LIFEJACKET_POSITIVE_ALIASES = [
    "life jacket",
    "lifejacket",
    "life_jacket",
    "life-jacket",
    "life jacket on",
    "lifejacket_on",
    "wear",
    "wearing",
    "with_lifejacket",
    "with life jacket",
    "life vest",
    "lifevest",
    "life_vest",
    "life-vest",
    "vest",
    "safety vest",
    "safety_vest",
    "safety-vest",
    "safetyvest",
]

# Aliases for negative torso PPE (NO Life Jacket / NO Vest)
LIFEJACKET_NEGATIVE_ALIASES = [
    "no life jacket",
    "no-life jacket",
    "no_life_jacket",
    "no-lifejacket",
    "nolifejacket",
    "life jacket off",
    "lifejacket_off",
    "notwear",
    "not wearing",
    "not_wearing",
    "without_lifejacket",
    "without life jacket",
    "no life vest",
    "no-life vest",
    "no_life_vest",
    "no lifevest",
    "no vest",
    "no-vest",
    "no safety vest",
    "no-safety vest",
    "no_safety_vest",
    "nosafetyvest",
]

# Aliases for Head PPE
HARDHAT_POSITIVE_ALIASES = [
    "hardhat",
    "hard hat",
    "hard-hat",
    "helmet",
    "safety helmet",
    "with_helmet",
    "with helmet",
    "wear_helmet",
]

HARDHAT_NEGATIVE_ALIASES = [
    "no hardhat",
    "no-hardhat",
    "no_hardhat",
    "no helmet",
    "no-helmet",
    "no_helmet",
    "without_helmet",
    "without helmet",
]

# Aliases for Person
PERSON_ALIASES = ["person", "human", "worker", "people", "pedestrian"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Roboflow Download
# ---------------------------------------------------------------------------


def download_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    download_dir: Path,
    format: str = "yolov8",
) -> Path:
    """Download a dataset from Roboflow Universe.

    Args:
        api_key: Roboflow API key.
        workspace: Roboflow workspace ID.
        project: Roboflow project ID.
        version: Dataset version number.
        download_dir: Directory to download dataset to.
        format: Export format (default: yolov8).

    Returns:
        Path to the downloaded dataset root.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error(
            "roboflow package not installed. Install with: pip install roboflow"
        )
        sys.exit(1)

    logger.info(
        "Downloading from Roboflow: workspace=%s, project=%s, version=%d",
        workspace,
        project,
        version,
    )

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version)

    download_dir.mkdir(parents=True, exist_ok=True)
    dataset = ds.download(format, location=str(download_dir))

    dataset_path = Path(dataset.location)
    logger.info("Dataset downloaded to: %s", dataset_path)
    return dataset_path


# ---------------------------------------------------------------------------
# Class Mapping Builder
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Normalize a class name for comparison."""
    return name.strip().lower().replace("_", " ").replace("-", " ")


def build_class_mapping(
    source_names: List[str],
    positive_class: Optional[str] = None,
    negative_class: Optional[str] = None,
    as_safety_vest: bool = True,
) -> Dict[int, Tuple[int, str]]:
    """Build a mapping from source class IDs to target class IDs.

    When ``as_safety_vest`` is True (default):
      - Life Jacket positive aliases map to Class 3 ("Safety Vest")
      - Life Jacket negative aliases map to Class 4 ("NO-Safety Vest")
      - Target is the clean 5-class schema.

    When ``as_safety_vest`` is False:
      - Life Jacket positive aliases map to Class 5 ("Life Jacket")
      - Life Jacket negative aliases map to Class 6 ("NO-Life Jacket")
      - Target is the expanded 7-class schema.

    Args:
        source_names: List of class names from the source dataset.
        positive_class: Explicit positive class name override (e.g., "wear").
        negative_class: Explicit negative class name override (e.g., "notwear").
        as_safety_vest: Whether to map Life Jacket into Safety Vest (5-class mode).

    Returns:
        Dict mapping source_class_id -> (target_class_id, target_class_name).
    """
    target_names = TARGET_NAMES_5CLASS if as_safety_vest else TARGET_NAMES_7CLASS
    pos_target_id = 3 if as_safety_vest else 5
    pos_target_name = "Safety Vest" if as_safety_vest else "Life Jacket"
    neg_target_id = 4 if as_safety_vest else 6
    neg_target_name = "NO-Safety Vest" if as_safety_vest else "NO-Life Jacket"

    mapping: Dict[int, Tuple[int, str]] = {}

    for src_idx, src_name in enumerate(source_names):
        src_norm = _normalize(src_name)

        # 1. Explicit override check
        if positive_class and src_norm == _normalize(positive_class):
            mapping[src_idx] = (pos_target_id, pos_target_name)
            continue
        if negative_class and src_norm == _normalize(negative_class):
            mapping[src_idx] = (neg_target_id, neg_target_name)
            continue

        # 2. Check standard 5 PPE class exact matches
        matched_standard = False
        for tgt_idx, tgt_name in enumerate(TARGET_NAMES_5CLASS):
            if src_norm == _normalize(tgt_name):
                mapping[src_idx] = (tgt_idx, tgt_name)
                matched_standard = True
                break
        if matched_standard:
            continue

        # 3. Check Life Jacket aliases
        if any(src_norm == _normalize(a) for a in LIFEJACKET_POSITIVE_ALIASES):
            mapping[src_idx] = (pos_target_id, pos_target_name)
        elif any(src_norm == _normalize(a) for a in LIFEJACKET_NEGATIVE_ALIASES):
            mapping[src_idx] = (neg_target_id, neg_target_name)

        # 4. Check Hardhat aliases
        elif any(src_norm == _normalize(a) for a in HARDHAT_POSITIVE_ALIASES):
            mapping[src_idx] = (1, "Hardhat")
        elif any(src_norm == _normalize(a) for a in HARDHAT_NEGATIVE_ALIASES):
            mapping[src_idx] = (2, "NO-Hardhat")

        # 5. Check Person aliases
        elif any(src_norm == _normalize(a) for a in PERSON_ALIASES):
            mapping[src_idx] = (0, "Person")

        else:
            logger.warning(
                "Source class '%s' (ID %d) did not match any PPE class — it will be SKIPPED.",
                src_name,
                src_idx,
            )

    return mapping


# ---------------------------------------------------------------------------
# Dataset Merger
# ---------------------------------------------------------------------------


def merge_lifejacket_dataset(
    source_dir: Path,
    existing_dataset_dir: Path,
    output_dir: Path,
    positive_class: Optional[str] = None,
    negative_class: Optional[str] = None,
    as_safety_vest: bool = True,
) -> Path:
    """Merge a life jacket source dataset into the existing PPE dataset.

    Args:
        source_dir: Root of the life jacket source dataset (must contain data.yaml).
        existing_dataset_dir: Root of the existing 5-class PPE dataset.
        output_dir: Output directory for the merged dataset.
        positive_class: Override for the positive life jacket class name.
        negative_class: Override for the negative life jacket class name.
        as_safety_vest: If True, map life jacket into Safety Vest (5-class mode).

    Returns:
        Path to the generated data.yaml in the output directory.
    """
    import yaml

    target_names = TARGET_NAMES_5CLASS if as_safety_vest else TARGET_NAMES_7CLASS

    # ---- Parse source dataset ----
    src_yaml_path = source_dir / "data.yaml"
    if not src_yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found in source dataset: {src_yaml_path}")

    with open(src_yaml_path, "r", encoding="utf-8") as f:
        src_yaml = yaml.safe_load(f)

    src_names: List[str] = src_yaml.get("names", [])
    if isinstance(src_names, dict):
        src_names = [src_names[k] for k in sorted(src_names.keys())]

    logger.info("Source dataset classes (%d): %s", len(src_names), src_names)
    mode_desc = (
        "5-Class Mode (Life Jacket mapped to 'Safety Vest')"
        if as_safety_vest
        else "7-Class Mode (Dedicated 'Life Jacket' classes)"
    )
    logger.info("Target Mode: %s", mode_desc)

    # Build mapping
    class_mapping = build_class_mapping(
        source_names=src_names,
        positive_class=positive_class,
        negative_class=negative_class,
        as_safety_vest=as_safety_vest,
    )

    logger.info("Class mapping (source -> target):")
    for src_id, (tgt_id, tgt_name) in sorted(class_mapping.items()):
        logger.info("  %d ('%s') -> %d ('%s')", src_id, src_names[src_id], tgt_id, tgt_name)

    # Verify at least one torso class mapped
    mapped_target_ids = {v[0] for v in class_mapping.values()}
    required_pos = 3 if as_safety_vest else 5
    required_neg = 4 if as_safety_vest else 6
    if required_pos not in mapped_target_ids and required_neg not in mapped_target_ids:
        logger.error(
            "No life jacket or torso PPE classes found in source dataset! "
            "Available classes: %s. Use --positive-class / --negative-class to specify.",
            src_names,
        )
        sys.exit(1)

    # ---- Setup output directory ----
    out_root = output_dir.resolve()
    is_in_place = out_root == existing_dataset_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    stats: Dict[str, Dict[str, int]] = {
        s: {
            "copied_existing": 0,
            "added_new": 0,
            "new_instances": 0,
            "pos_vest_instances": 0,
            "neg_vest_instances": 0,
        }
        for s in splits
    }

    for split in splits:
        out_img_dir = out_root / split / "images"
        out_lbl_dir = out_root / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: Copy existing dataset (if not operating in-place) ----
        if not is_in_place:
            existing_img_dir = existing_dataset_dir / split / "images"
            existing_lbl_dir = existing_dataset_dir / split / "labels"

            if existing_img_dir.is_dir():
                for img_file in existing_img_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                        dst = out_img_dir / img_file.name
                        if not dst.exists():
                            try:
                                os.link(str(img_file), str(dst))
                            except Exception:
                                shutil.copy2(str(img_file), str(dst))
                        stats[split]["copied_existing"] += 1

            if existing_lbl_dir.is_dir():
                for lbl_file in existing_lbl_dir.iterdir():
                    if lbl_file.is_file() and lbl_file.suffix == ".txt":
                        dst = out_lbl_dir / lbl_file.name
                        if not dst.exists():
                            shutil.copy2(str(lbl_file), str(dst))
        else:
            # When in-place, count existing images
            existing_img_dir = out_img_dir
            if existing_img_dir.is_dir():
                stats[split]["copied_existing"] = sum(
                    1
                    for f in existing_img_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                )

        # ---- Step 2: Merge life jacket dataset ----
        raw_split_key = "val" if split == "valid" and "val" in src_yaml else split
        if split == "valid" and raw_split_key not in src_yaml:
            raw_split_key = "valid"

        raw_split_path = src_yaml.get(raw_split_key, "")
        if not raw_split_path:
            logger.warning("Split '%s' not found in source data.yaml - skipping.", split)
            continue

        src_img_dir = _resolve_source_split(source_dir, raw_split_path)
        if src_img_dir is None:
            logger.warning("Could not find split '%s' images in source dataset.", split)
            continue

        src_lbl_dir = src_img_dir.parent / "labels"
        if not src_lbl_dir.is_dir():
            for candidate in [
                source_dir / split / "labels",
                source_dir / raw_split_key / "labels",
            ]:
                if candidate.is_dir():
                    src_lbl_dir = candidate
                    break

        if not src_lbl_dir.is_dir():
            logger.warning("Labels directory not found for split '%s'.", split)
            continue

        logger.info(
            "Merging split '%s': images=%s, labels=%s",
            split,
            src_img_dir,
            src_lbl_dir,
        )

        img_files = [
            f
            for f in src_img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for img_path in img_files:
            lbl_name = img_path.stem + ".txt"
            src_lbl_path = src_lbl_dir / lbl_name

            remapped_lines: List[str] = []
            pos_count = 0
            neg_count = 0

            if src_lbl_path.is_file():
                with open(src_lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                orig_cls = int(parts[0])
                                if orig_cls in class_mapping:
                                    new_cls = class_mapping[orig_cls][0]
                                    remapped_lines.append(
                                        f"{new_cls} {' '.join(parts[1:])}\n"
                                    )
                                    if new_cls == required_pos:
                                        pos_count += 1
                                    elif new_cls == required_neg:
                                        neg_count += 1
                            except ValueError:
                                continue

            if not remapped_lines:
                continue  # Skip images with no relevant annotations

            # Write remapped label (prefix with "lj_" to avoid name collisions)
            safe_name = f"lj_{img_path.stem}"
            out_lbl_path = out_lbl_dir / f"{safe_name}.txt"
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.writelines(remapped_lines)

            # Copy image
            out_img_path = out_img_dir / f"{safe_name}{img_path.suffix}"
            if not out_img_path.exists():
                try:
                    os.link(str(img_path), str(out_img_path))
                except Exception:
                    shutil.copy2(str(img_path), str(out_img_path))

            stats[split]["added_new"] += 1
            stats[split]["new_instances"] += len(remapped_lines)
            stats[split]["pos_vest_instances"] += pos_count
            stats[split]["neg_vest_instances"] += neg_count

    # ---- Write new data.yaml ----
    new_yaml = {
        "path": str(out_root).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(target_names),
        "names": target_names,
    }

    out_yaml_path = out_root / "data.yaml"
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(new_yaml, f, sort_keys=False)

    # ---- Report ----
    print("\n" + "=" * 70)
    print("  PPE DATASET MERGE - COMPLETE")
    print("=" * 70)
    print(f"  Output Directory : {out_root}")
    print(f"  Mode             : {mode_desc}")
    print(f"  Total Classes    : {len(target_names)}")
    for i, name in enumerate(target_names):
        marker = "★" if i == 3 else "•"
        tag = " (Life Jacket mapped here)" if as_safety_vest and i == 3 else ""
        print(f"    {marker} {i}: {name}{tag}")
    print()

    for s in splits:
        st = stats[s]
        total = st["copied_existing"] + st["added_new"]
        print(
            f"  [{s.upper():>5s}]  existing: {st['copied_existing']:>5d}  "
            f"+ new: {st['added_new']:>5d}  = total: {total:>5d} images  "
            f"(new instances: {st['new_instances']:>4d} | "
            f"+vest: {st['pos_vest_instances']:>3d}, -vest: {st['neg_vest_instances']:>3d})"
        )

    print("=" * 70 + "\n")
    logger.info("Merged data.yaml written to: %s", out_yaml_path)

    return out_yaml_path


def _resolve_source_split(dataset_root: Path, raw_path: str) -> Optional[Path]:
    """Resolve a split images directory from a data.yaml path entry."""
    candidate = (dataset_root / raw_path).resolve()
    if candidate.is_dir():
        return candidate

    parts = Path(raw_path.replace("\\", "/")).parts
    cleaned = [p for p in parts if p != ".."]
    if cleaned:
        fallback = dataset_root.joinpath(*cleaned)
        if fallback.is_dir():
            return fallback

    for suffix in ["/images", ""]:
        for split_name in ["train", "valid", "val", "test"]:
            cand = dataset_root / f"{split_name}{suffix}"
            if cand.is_dir():
                return cand

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a Life Jacket dataset into the PPE compliance dataset.\n"
            "By default, maps Life Jacket images to 'Safety Vest' so the model detects\n"
            "life jackets but reports them as Safety Vest for full frontend/backend compatibility."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge local dataset into a new augmented 5-class dataset:
  python scripts/add_lifejacket_dataset.py --source "C:/path/to/lifejacket-dataset"

  # Merge directly in-place into existing data/processed/ppe_5class:
  python scripts/add_lifejacket_dataset.py --source "C:/path/to/lifejacket-dataset" --in-place

  # Download from Roboflow Universe:
  python scripts/add_lifejacket_dataset.py \\
      --api-key YOUR_KEY --workspace contil-rn9fi \\
      --project life-jacket-kt521 --version 1

  # Custom positive/negative class names:
  python scripts/add_lifejacket_dataset.py --source "C:/path/to/dataset" \\
      --positive-class "wear" --negative-class "notwear"
        """,
    )

    # Source options
    source_group = parser.add_argument_group("Source Dataset")
    source_group.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a local life jacket dataset in YOLO format.",
    )

    # Roboflow options
    rf_group = parser.add_argument_group("Roboflow Download")
    rf_group.add_argument("--api-key", type=str, default=None, help="Roboflow API key.")
    rf_group.add_argument(
        "--workspace", type=str, default=None, help="Roboflow workspace ID."
    )
    rf_group.add_argument(
        "--project", type=str, default=None, help="Roboflow project ID."
    )
    rf_group.add_argument(
        "--version", type=int, default=1, help="Roboflow dataset version (default: 1)."
    )

    # Class mapping options
    cls_group = parser.add_argument_group("Class Mapping & Mode")
    cls_group.add_argument(
        "--positive-class",
        type=str,
        default=None,
        help="Source class name for Life Jacket (e.g. 'wear', 'Life Jacket ON').",
    )
    cls_group.add_argument(
        "--negative-class",
        type=str,
        default=None,
        help="Source class name for NO-Life Jacket (e.g. 'notwear', 'Life Jacket OFF').",
    )
    cls_group.add_argument(
        "--separate-classes",
        action="store_true",
        default=False,
        help="Produce a 7-class dataset with dedicated 'Life Jacket' classes instead of mapping to 'Safety Vest'.",
    )

    # Paths
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--existing-dataset",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "processed" / "ppe_5class"),
        help="Path to existing 5-class PPE dataset (default: data/processed/ppe_5class).",
    )
    path_group.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "processed" / "ppe_5class_augmented"),
        help="Output directory for merged dataset (default: data/processed/ppe_5class_augmented).",
    )
    path_group.add_argument(
        "--in-place",
        action="store_true",
        default=False,
        help="Merge directly into --existing-dataset instead of creating a new output folder.",
    )
    path_group.add_argument(
        "--download-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "raw" / "lifejacket_roboflow"),
        help="Directory for Roboflow downloads (default: data/raw/lifejacket_roboflow).",
    )

    args = parser.parse_args()

    # Determine source directory
    if args.source:
        source_dir = Path(args.source).resolve()
        if not source_dir.is_dir():
            logger.error("Source directory does not exist: %s", source_dir)
            sys.exit(1)
    elif args.api_key and args.workspace and args.project:
        source_dir = download_from_roboflow(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            download_dir=Path(args.download_dir),
        )
    else:
        parser.error(
            "Provide either --source (local dataset) or "
            "--api-key + --workspace + --project (Roboflow download)."
        )
        return

    existing_dir = Path(args.existing_dataset).resolve()
    if not existing_dir.is_dir():
        logger.error("Existing dataset not found: %s", existing_dir)
        sys.exit(1)

    output_dir = existing_dir if args.in_place else Path(args.output).resolve()
    as_safety_vest = not args.separate_classes

    merge_lifejacket_dataset(
        source_dir=source_dir,
        existing_dataset_dir=existing_dir,
        output_dir=output_dir,
        positive_class=args.positive_class,
        negative_class=args.negative_class,
        as_safety_vest=as_safety_vest,
    )


if __name__ == "__main__":
    main()
