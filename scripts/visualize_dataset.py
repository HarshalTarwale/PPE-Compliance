"""
Dataset visualization utility for the PPE Compliance project.

Renders random training/validation/test images with their ground-truth
YOLO bounding boxes and class names overlaid.

Usage:
    python scripts/visualize_dataset.py <dataset-path>
    python scripts/visualize_dataset.py <dataset-path> --split train --count 5
    python scripts/visualize_dataset.py <dataset-path> --save-dir outputs/viz

Source images are NEVER modified.  Visualizations are either displayed
in an OpenCV window or saved to the specified directory.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_logger  # noqa: E402
from scripts.check_dataset import (  # noqa: E402
    IMAGE_EXTENSIONS,
    parse_data_yaml,
    resolve_split_dirs,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Color palette — visually distinct, easy on the eyes
# ---------------------------------------------------------------------------

def _generate_palette(n: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colors using the HSV wheel."""
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))  # OpenCV hue range is [0, 180)
        hsv = np.uint8([[[hue, 200, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


# ---------------------------------------------------------------------------
# YOLO → pixel coordinate conversion
# ---------------------------------------------------------------------------

def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Convert normalised YOLO (x_c, y_c, w, h) to pixel (x1, y1, x2, y2)."""
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    return (
        max(0, x1),
        max(0, y1),
        min(img_w, x2),
        min(img_h, y2),
    )


# ---------------------------------------------------------------------------
# Label parsing (tolerant of malformed lines)
# ---------------------------------------------------------------------------

def parse_labels(
    label_path: Path,
    image_name: str,
) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO label file, skipping malformed lines.

    Returns list of (class_id, x_center, y_center, width, height).
    """
    annotations: List[Tuple[int, float, float, float, float]] = []

    if not label_path.is_file():
        logger.warning("Label file not found: %s", label_path)
        return annotations

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Cannot read %s: %s", label_path.name, exc)
        return annotations

    if not text:
        logger.info("%s — empty label (background-only image)", image_name)
        return annotations

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            logger.warning(
                "%s line %d: expected ≥5 fields, got %d — skipping",
                image_name, line_no, len(parts),
            )
            continue
        try:
            cls_id = int(parts[0])
            coords = tuple(float(v) for v in parts[1:5])
        except ValueError:
            logger.warning(
                "%s line %d: non-numeric values — skipping", image_name, line_no
            )
            continue

        annotations.append((cls_id, *coords))  # type: ignore[arg-type]

    return annotations


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2


def draw_annotations(
    image: np.ndarray,
    annotations: List[Tuple[int, float, float, float, float]],
    class_names: Dict[int, str],
    palette: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Draw bounding boxes and class labels on a COPY of the image.

    The original ``image`` array is never mutated.
    """
    canvas = image.copy()
    img_h, img_w = canvas.shape[:2]

    for cls_id, xc, yc, bw, bh in annotations:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h)
        color = palette[cls_id % len(palette)]
        label = class_names.get(cls_id, f"cls_{cls_id}")

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(
            canvas,
            (x1, label_y1),
            (x1 + tw + 4, label_y1 + th + baseline + 4),
            color,
            cv2.FILLED,
        )

        # Label text — pick white or black for contrast
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_y1 + th + 2),
            _FONT,
            _FONT_SCALE,
            text_color,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Core visualisation routine
# ---------------------------------------------------------------------------


def visualize_samples(
    dataset_root: Path,
    split: str = "train",
    count: int = 5,
    save_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    max_display_width: int = 1280,
) -> List[Path]:
    """Visualize random samples from a dataset split.

    Args:
        dataset_root: Path to dataset root (must contain data.yaml).
        split: One of "train", "valid", "test".
        count: Number of random images to visualize.
        save_dir: If set, save annotated images here instead of displaying.
        seed: Random seed for reproducibility (None = non-deterministic).
        max_display_width: Resize for display if image is wider than this.

    Returns:
        List of saved image paths (empty when displaying interactively).
    """
    # --- Load YAML ---
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.is_file():
        logger.error("data.yaml not found at %s", yaml_path)
        sys.exit(1)

    yaml_data = parse_data_yaml(yaml_path)

    # Class names
    names_raw = yaml_data.get("names", {})
    if isinstance(names_raw, dict):
        class_names: Dict[int, str] = {int(k): str(v) for k, v in names_raw.items()}
    elif isinstance(names_raw, list):
        class_names = {i: str(v) for i, v in enumerate(names_raw)}
    else:
        logger.error("Cannot parse 'names' from data.yaml")
        sys.exit(1)

    num_classes = yaml_data.get("nc", len(class_names))
    palette = _generate_palette(num_classes)

    # --- Resolve split ---
    # Normalize "val" → "valid" for the resolver
    split_key = "valid" if split == "val" else split
    split_dirs = resolve_split_dirs(dataset_root, yaml_data)

    if split_key not in split_dirs:
        logger.error("Split '%s' not found.  Available: %s", split, list(split_dirs.keys()))
        sys.exit(1)

    split_dir = split_dirs[split_key]
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    # --- Collect images ---
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.error("No images found in %s", images_dir)
        sys.exit(1)

    count = min(count, len(image_files))
    if seed is not None:
        random.seed(seed)
    selected = random.sample(image_files, count)

    logger.info(
        "Selected %d/%d images from [%s] split",
        count, len(image_files), split,
    )

    # --- Save dir setup ---
    saved_paths: List[Path] = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each image ---
    for idx, img_path in enumerate(selected, start=1):
        print(f"  [{idx}/{count}] {img_path.name}")

        # Read image (never modified)
        raw = cv2.imread(str(img_path))
        if raw is None:
            logger.warning("Cannot read image: %s — skipping", img_path.name)
            continue

        # Load labels
        label_path = labels_dir / (img_path.stem + ".txt")
        annotations = parse_labels(label_path, img_path.name)

        # Draw on a copy
        annotated = draw_annotations(raw, annotations, class_names, palette)

        if save_dir is not None:
            out_path = save_dir / f"viz_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), annotated)
            saved_paths.append(out_path)
            logger.info("Saved: %s", out_path)
        else:
            # Resize for display if too large
            h, w = annotated.shape[:2]
            if w > max_display_width:
                scale = max_display_width / w
                annotated = cv2.resize(
                    annotated, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            window_title = f"[{idx}/{count}] {img_path.name}  |  Press any key / Q to quit"
            cv2.imshow(window_title, annotated)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key in (ord("q"), ord("Q"), 27):  # q or Esc
                print("  Stopped by user.")
                break

    if saved_paths:
        print(f"\n  ✔ Saved {len(saved_paths)} visualization(s) to: {save_dir}")

    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize random dataset images with ground-truth YOLO bounding boxes.",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory (must contain data.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "valid", "test"],
        help="Dataset split to visualize (default: train).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of random images to visualize (default: 5).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save annotated images.  If omitted, images are "
             "displayed in an OpenCV window.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path).resolve()
    if not dataset_root.is_dir():
        logger.error("Dataset path does not exist: %s", dataset_root)
        sys.exit(1)

    save_dir = Path(args.save_dir) if args.save_dir else None

    print(f"\n  Dataset : {dataset_root}")
    print(f"  Split   : {args.split}")
    print(f"  Count   : {args.count}")
    print(f"  Output  : {save_dir or 'interactive display'}\n")

    visualize_samples(
        dataset_root=dataset_root,
        split=args.split,
        count=args.count,
        save_dir=save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
