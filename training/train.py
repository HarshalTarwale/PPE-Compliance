"""
YOLOv8n PPE detection training pipeline.

Trains a YOLOv8n model on the Construction Site Safety dataset using
the Ultralytics library.  All training parameters are configurable via
command-line arguments.

Usage:
    python training/train.py --data <path-to-data.yaml>
    python training/train.py --data <path-to-data.yaml> --epochs 100 --batch 8 --device cpu

The script validates the dataset path, fixes common Roboflow path
issues in data.yaml, loads pretrained YOLOv8n weights, and saves
results (weights, metrics, config) to an organized experiment directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import Settings, get_logger  # noqa: E402
from scripts.check_dataset import parse_data_yaml  # noqa: E402

logger = get_logger(__name__)
settings = Settings()


# ---------------------------------------------------------------------------
# data.yaml path fixer
# ---------------------------------------------------------------------------


def _resolve_yaml_path(yaml_dir: Path, raw_path: str) -> str:
    """Resolve a single path entry from data.yaml.

    Roboflow exports often use ``../train/images`` which breaks when
    the YAML is already at the dataset root.  This function checks
    whether the literal path resolves; if not, it strips leading ``../``
    and tries again relative to the YAML directory.

    Returns the corrected *absolute* path string.
    """
    candidate = (yaml_dir / raw_path).resolve()
    if candidate.exists():
        return str(candidate)

    # Strip leading "../" segments and retry
    parts = Path(raw_path.replace("\\", "/")).parts
    cleaned = [p for p in parts if p != ".."]
    if cleaned:
        fallback = (yaml_dir / Path(*cleaned)).resolve()
        if fallback.exists():
            return str(fallback)

    # Return original if nothing works — YOLO will error clearly
    return str(candidate)


def prepare_data_yaml(original_yaml: Path, output_dir: Path) -> Path:
    """Create a copy of data.yaml with corrected absolute paths.

    The corrected YAML is saved to ``output_dir/data.yaml`` so the
    original is never modified.

    Args:
        original_yaml: Path to the original data.yaml.
        output_dir: Directory to write the fixed copy.

    Returns:
        Path to the corrected data.yaml.
    """
    yaml_data = parse_data_yaml(original_yaml)
    yaml_dir = original_yaml.parent

    # Fix split paths
    for key in ("train", "val", "valid", "test"):
        if key in yaml_data:
            yaml_data[key] = _resolve_yaml_path(yaml_dir, yaml_data[key])

    # Write corrected YAML
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_yaml = output_dir / "data.yaml"

    try:
        import yaml  # type: ignore

        with open(fixed_yaml, "w", encoding="utf-8") as fh:
            yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback: write manually
        _write_yaml_fallback(yaml_data, fixed_yaml)

    logger.info("Corrected data.yaml saved to: %s", fixed_yaml)
    return fixed_yaml


def _write_yaml_fallback(data: Dict[str, Any], path: Path) -> None:
    """Write a minimal YAML file without PyYAML dependency."""
    lines = []
    for key, value in data.items():
        if key == "names":
            lines.append("names:")
            if isinstance(value, dict):
                for idx in sorted(value.keys()):
                    lines.append(f"  {idx}: {value[idx]}")
            elif isinstance(value, list):
                for idx, name in enumerate(value):
                    lines.append(f"  {idx}: {name}")
        elif key == "roboflow":
            # Preserve roboflow metadata block
            lines.append("roboflow:")
            if isinstance(value, dict):
                for rk, rv in value.items():
                    lines.append(f"  {rk}: {rv}")
        else:
            lines.append(f"{key}: {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def validate_dataset(data_yaml: Path) -> Dict[str, Any]:
    """Validate that the dataset path exists and data.yaml is parseable.

    Returns the parsed YAML data.

    Raises:
        SystemExit: If validation fails.
    """
    if not data_yaml.is_file():
        logger.error("data.yaml not found: %s", data_yaml)
        sys.exit(1)

    yaml_data = parse_data_yaml(data_yaml)

    nc = yaml_data.get("nc")
    names = yaml_data.get("names")
    if nc is None or names is None:
        logger.error("data.yaml is missing 'nc' or 'names' field")
        sys.exit(1)

    # Check at least train split is referenced
    if "train" not in yaml_data:
        logger.error("data.yaml is missing 'train' path")
        sys.exit(1)

    logger.info("Dataset validated: %d classes", nc)
    return yaml_data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_yaml: Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project: str = "runs/train",
    name: str = "ppe_yolov8n",
    model_weights: str = "yolov8n.pt",
    resume: bool = False,
) -> Path:
    """Train a YOLOv8n model on the PPE dataset.

    Args:
        data_yaml: Path to the (corrected) data.yaml.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        device: Device string — "auto", "cpu", "0", "0,1", etc.
        project: Project directory for experiment outputs.
        name: Experiment name.
        model_weights: Pretrained weights to start from.
        resume: Whether to resume from last checkpoint.

    Returns:
        Path to the experiment output directory.
    """
    from ultralytics import YOLO  # type: ignore

    # Resolve device
    if device == "auto":
        import torch  # type: ignore

        resolved_device = "0" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Auto-detected device: %s (CUDA available: %s)",
            resolved_device,
            torch.cuda.is_available(),
        )
    else:
        resolved_device = device

    # Resolve project path
    project_path = (
        Path(project)
        if Path(project).is_absolute()
        else _PROJECT_ROOT / project
    )

    # Print config
    config = {
        "model_weights": model_weights,
        "data_yaml": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": resolved_device,
        "project": str(project_path),
        "name": name,
        "resume": resume,
        "timestamp": datetime.now().isoformat(),
    }

    print("\n" + "=" * 60)
    print("  YOLOv8n PPE DETECTION — TRAINING CONFIG")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:<18s}: {v}")
    print("=" * 60 + "\n")

    # Load model
    logger.info("Loading model: %s", model_weights)
    model = YOLO(model_weights)

    # Train
    logger.info("Starting training for %d epochs …", epochs)
    t0 = time.time()

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=str(project_path),
        name=name,
        exist_ok=True,
        pretrained=True,
        save=True,
        save_period=-1,      # Save only best and last
        plots=True,          # Generate training plots
        verbose=True,
    )

    elapsed = time.time() - t0
    config["training_time_seconds"] = round(elapsed, 2)
    config["training_time_human"] = _format_duration(elapsed)

    # Save experiment config as JSON
    exp_dir = project_path / name
    config_path = exp_dir / "train_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Experiment config saved to: %s", config_path)

    # Report
    best_weights = exp_dir / "weights" / "best.pt"
    last_weights = exp_dir / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Duration     : {config['training_time_human']}")
    print(f"  Experiment   : {exp_dir}")
    print(f"  Best weights : {best_weights}")
    print(f"  Last weights : {last_weights}")
    print(f"  Config       : {config_path}")
    if best_weights.exists():
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        print(f"  Best size    : {size_mb:.1f} MB")
    print("=" * 60 + "\n")

    return exp_dir


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n on the PPE detection dataset.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml (dataset root must contain train/valid/test splits).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16). Use -1 for auto-batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device: "auto", "cpu", "0", "0,1" etc. (default: auto).',
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory for outputs (default: runs/train).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ppe_yolov8n",
        help="Experiment name (default: ppe_yolov8n).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from last checkpoint.",
    )
    args = parser.parse_args()

    # --- Validate dataset ---
    data_path = Path(args.data).resolve()

    # Accept either a data.yaml file or its parent directory
    if data_path.is_dir():
        data_yaml = data_path / "data.yaml"
    else:
        data_yaml = data_path

    validate_dataset(data_yaml)

    # --- Prepare corrected data.yaml ---
    project_path = (
        Path(args.project)
        if Path(args.project).is_absolute()
        else _PROJECT_ROOT / args.project
    )
    fixed_yaml = prepare_data_yaml(data_yaml, project_path / args.name)

    # --- Train ---
    train(
        data_yaml=fixed_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        model_weights=args.weights,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
