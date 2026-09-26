"""
Dataset inspection and validation pipeline for the PPE Compliance project.

Validates the Construction Site Safety dataset (YOLOv8 format):
  - Dataset structure (splits, image/label dirs)
  - Image existence and readability
  - Label existence and format
  - Bounding-box value ranges
  - Class-ID validity (read from data.yaml, NOT hardcoded)
  - Missing or empty annotation files
  - Corrupt / unreadable images
  - Class distribution statistics
  - Image resolution statistics
  - Summary report (terminal + JSON)

Usage:
    python scripts/check_dataset.py <path-to-dataset-root>
    python scripts/check_dataset.py <path-to-dataset-root> --json

The dataset root must contain a ``data.yaml`` file and split directories
(train, valid, test) each with ``images/`` and ``labels/`` sub-folders.

This script does NOT modify or delete any files.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import Settings, get_logger  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_CLASSES: List[str] = [
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
    "Person",
]
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
EXPECTED_SPLITS: List[str] = ["train", "valid", "test"]

# ---------------------------------------------------------------------------
# Data-classes for the report
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    """A single validation issue."""

    split: str
    file: str
    issue_type: str
    detail: str


@dataclass
class SplitStats:
    """Aggregated statistics for one dataset split."""

    image_count: int = 0
    label_count: int = 0
    images_without_labels: int = 0
    labels_without_images: int = 0
    empty_labels: int = 0
    corrupt_images: int = 0
    invalid_bbox_count: int = 0
    invalid_class_id_count: int = 0
    total_annotations: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    widths: List[int] = field(default_factory=list)
    heights: List[int] = field(default_factory=list)


@dataclass
class DatasetReport:
    """Complete validation report."""

    dataset_path: str = ""
    data_yaml_found: bool = False
    num_classes_yaml: int = 0
    class_names_yaml: Dict[int, str] = field(default_factory=dict)
    required_classes_found: List[str] = field(default_factory=list)
    required_classes_missing: List[str] = field(default_factory=list)
    splits_found: List[str] = field(default_factory=list)
    splits_missing: List[str] = field(default_factory=list)
    split_stats: Dict[str, SplitStats] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    passed: bool = False


# ---------------------------------------------------------------------------
# YAML parsing (minimal — avoids heavy dependency if PyYAML missing)
# ---------------------------------------------------------------------------


def parse_data_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Parse a YOLOv8 ``data.yaml`` file.

    Tries PyYAML first; falls back to a minimal hand-parser for the
    simple key-value structure YOLOv8 uses.
    """
    try:
        import yaml  # type: ignore

        with open(yaml_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        logger.warning("PyYAML not installed — using fallback YAML parser")
        return _fallback_yaml_parse(yaml_path)


def _fallback_yaml_parse(yaml_path: Path) -> Dict[str, Any]:
    """Very basic parser for flat YOLOv8 data.yaml files."""
    data: Dict[str, Any] = {}
    names: Dict[int, str] = {}
    in_names = False

    with open(yaml_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Detect start of names block
            if line.startswith("names:"):
                in_names = True
                # Inline list style: names: ['a', 'b']
                rest = line.split(":", 1)[1].strip()
                if rest.startswith("["):
                    items = rest.strip("[]").split(",")
                    for idx, item in enumerate(items):
                        names[idx] = item.strip().strip("'\"")
                    in_names = False
                continue

            if in_names:
                # "  0: Hardhat" style
                if ":" in line:
                    parts = line.split(":", 1)
                    try:
                        idx = int(parts[0].strip())
                        names[idx] = parts[1].strip().strip("'\"")
                    except ValueError:
                        in_names = False
                elif line.startswith("-"):
                    # "- Hardhat" list style
                    idx = len(names)
                    names[idx] = line.lstrip("- ").strip("'\"")
                else:
                    in_names = False
            else:
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    # Try int
                    try:
                        data[key] = int(val)
                    except ValueError:
                        data[key] = val

    if names:
        data["names"] = names
    return data


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def resolve_split_dirs(
    dataset_root: Path, yaml_data: Dict[str, Any]
) -> Dict[str, Path]:
    """Resolve split paths from data.yaml or standard conventions.

    YOLOv8 data.yaml often uses keys like ``train``, ``val``/``valid``,
    ``test`` whose values are relative paths to the images directory.
    """
    splits: Dict[str, Path] = {}

    # Mapping: YAML key → canonical split name
    key_map = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

    for yaml_key, split_name in key_map.items():
        if yaml_key in yaml_data:
            raw = yaml_data[yaml_key]
            # The value may be e.g. "../train/images" or "./train/images"
            # Resolve relative to the directory containing data.yaml
            candidate = (dataset_root / raw).resolve()
            if not candidate.exists():
                # Roboflow exports often use "../train/images" which
                # resolves outside the dataset root.  Fall back to
                # looking for the tail components inside the root.
                # e.g. "../train/images" → try "<root>/train/images"
                tail = Path(raw.replace("\\", "/").lstrip("./"))
                # Strip leading ".." segments
                parts = tail.parts
                while parts and parts[0] == "..":
                    parts = parts[1:]
                if parts:
                    candidate = (dataset_root / Path(*parts)).resolve()
            # Go up from images/ to get the split root
            if candidate.exists() and candidate.name == "images":
                splits[split_name] = candidate.parent
            elif candidate.exists():
                splits[split_name] = candidate
        else:
            # Fall back to conventional directory names
            for name_variant in [split_name, split_name.replace("valid", "val")]:
                candidate = dataset_root / name_variant
                if candidate.is_dir():
                    splits[split_name] = candidate
                    break

    return splits


def validate_image(image_path: Path) -> Optional[Tuple[int, int]]:
    """Try to open an image and return (width, height), or None if corrupt."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img.verify()
        # Re-open because verify() may consume the stream
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except ImportError:
        # Pillow not available — skip corruption check, guess from file
        logger.warning("Pillow not installed — skipping image corruption checks")
        return (0, 0)
    except Exception:
        return None


def parse_label_file(
    label_path: Path,
    num_classes: int,
    split_name: str,
    issues: List[Issue],
) -> Tuple[List[int], int, int]:
    """Parse a YOLO-format label file.

    Returns:
        (class_ids, invalid_bbox_count, invalid_class_count)
    """
    class_ids: List[int] = []
    bad_bbox = 0
    bad_cls = 0

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        issues.append(
            Issue(split_name, label_path.name, "unreadable_label", str(exc))
        )
        return class_ids, bad_bbox, bad_cls

    if not text:
        return class_ids, bad_bbox, bad_cls  # empty file handled by caller

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            issues.append(
                Issue(
                    split_name,
                    label_path.name,
                    "malformed_line",
                    f"Line {line_no}: expected ≥5 fields, got {len(parts)}",
                )
            )
            continue

        # Class ID
        try:
            cls_id = int(parts[0])
        except ValueError:
            issues.append(
                Issue(
                    split_name,
                    label_path.name,
                    "invalid_class_id",
                    f"Line {line_no}: non-integer class ID '{parts[0]}'",
                )
            )
            bad_cls += 1
            continue

        if cls_id < 0 or cls_id >= num_classes:
            issues.append(
                Issue(
                    split_name,
                    label_path.name,
                    "class_id_out_of_range",
                    f"Line {line_no}: class_id={cls_id} not in [0, {num_classes})",
                )
            )
            bad_cls += 1
        class_ids.append(cls_id)

        # Bounding box (x_center, y_center, width, height) — all in [0, 1]
        try:
            coords = [float(c) for c in parts[1:5]]
        except ValueError:
            issues.append(
                Issue(
                    split_name,
                    label_path.name,
                    "non_numeric_bbox",
                    f"Line {line_no}: cannot parse bbox values",
                )
            )
            bad_bbox += 1
            continue

        for i, val in enumerate(coords):
            if val < 0.0 or val > 1.0:
                issues.append(
                    Issue(
                        split_name,
                        label_path.name,
                        "bbox_out_of_range",
                        f"Line {line_no}: bbox[{i}]={val:.4f} outside [0,1]",
                    )
                )
                bad_bbox += 1
                break  # one issue per line is enough

        # Width and height must be positive
        if coords[2] <= 0 or coords[3] <= 0:
            issues.append(
                Issue(
                    split_name,
                    label_path.name,
                    "bbox_zero_size",
                    f"Line {line_no}: w={coords[2]:.4f}, h={coords[3]:.4f}",
                )
            )
            bad_bbox += 1

    return class_ids, bad_bbox, bad_cls


# ---------------------------------------------------------------------------
# Per-split validation
# ---------------------------------------------------------------------------


def validate_split(
    split_name: str,
    split_dir: Path,
    num_classes: int,
    class_names: Dict[int, str],
    issues: List[Issue],
) -> SplitStats:
    """Validate all images and labels in a single split."""
    stats = SplitStats()

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir():
        issues.append(
            Issue(split_name, str(images_dir), "missing_dir", "images/ dir not found")
        )
        return stats

    if not labels_dir.is_dir():
        issues.append(
            Issue(split_name, str(labels_dir), "missing_dir", "labels/ dir not found")
        )
        return stats

    # Collect stems
    image_files = {
        f.stem: f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_files = {
        f.stem: f
        for f in labels_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    }

    stats.image_count = len(image_files)
    stats.label_count = len(label_files)

    # Images without labels
    images_only = set(image_files.keys()) - set(label_files.keys())
    stats.images_without_labels = len(images_only)
    for stem in sorted(images_only):
        issues.append(
            Issue(
                split_name,
                image_files[stem].name,
                "image_without_label",
                "No corresponding .txt label file",
            )
        )

    # Labels without images
    labels_only = set(label_files.keys()) - set(image_files.keys())
    stats.labels_without_images = len(labels_only)
    for stem in sorted(labels_only):
        issues.append(
            Issue(
                split_name,
                label_files[stem].name,
                "label_without_image",
                "No corresponding image file",
            )
        )

    # Validate each image + label pair
    class_counter: Counter = Counter()
    paired_stems = set(image_files.keys()) & set(label_files.keys())

    for stem in sorted(paired_stems):
        img_path = image_files[stem]
        lbl_path = label_files[stem]

        # --- Image validation ---
        result = validate_image(img_path)
        if result is None:
            stats.corrupt_images += 1
            issues.append(
                Issue(
                    split_name,
                    img_path.name,
                    "corrupt_image",
                    "Image cannot be opened or verified",
                )
            )
        else:
            w, h = result
            if w > 0 and h > 0:
                stats.widths.append(w)
                stats.heights.append(h)

        # --- Label validation ---
        lbl_text = ""
        try:
            lbl_text = lbl_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

        if not lbl_text:
            stats.empty_labels += 1
            issues.append(
                Issue(
                    split_name,
                    lbl_path.name,
                    "empty_label",
                    "Label file is empty (background-only image?)",
                )
            )
            continue

        cls_ids, bad_bbox, bad_cls = parse_label_file(
            lbl_path, num_classes, split_name, issues
        )
        stats.invalid_bbox_count += bad_bbox
        stats.invalid_class_id_count += bad_cls
        stats.total_annotations += len(cls_ids)

        for cid in cls_ids:
            name = class_names.get(cid, f"unknown_{cid}")
            class_counter[name] += 1

    stats.class_distribution = dict(class_counter.most_common())
    return stats


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------


def validate_dataset(dataset_root: Path) -> DatasetReport:
    """Run the full validation pipeline on a YOLOv8-format dataset."""
    report = DatasetReport(dataset_path=str(dataset_root))
    t0 = time.time()

    # ---- data.yaml ----
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.is_file():
        report.issues.append(
            Issue("", "data.yaml", "missing_file", "data.yaml not found in dataset root")
        )
        report.elapsed_seconds = time.time() - t0
        logger.error("data.yaml not found at %s", yaml_path)
        return report

    report.data_yaml_found = True
    yaml_data = parse_data_yaml(yaml_path)

    # Class names — read from YAML, never hardcode
    names_raw = yaml_data.get("names", {})
    if isinstance(names_raw, dict):
        report.class_names_yaml = {int(k): str(v) for k, v in names_raw.items()}
    elif isinstance(names_raw, list):
        report.class_names_yaml = {i: str(v) for i, v in enumerate(names_raw)}
    else:
        report.issues.append(
            Issue("", "data.yaml", "invalid_names", "Cannot parse 'names' field")
        )

    report.num_classes_yaml = yaml_data.get("nc", len(report.class_names_yaml))
    yaml_class_values = set(report.class_names_yaml.values())

    for req in REQUIRED_CLASSES:
        if req in yaml_class_values:
            report.required_classes_found.append(req)
        else:
            report.required_classes_missing.append(req)

    logger.info(
        "data.yaml: %d classes, required found: %d/%d",
        report.num_classes_yaml,
        len(report.required_classes_found),
        len(REQUIRED_CLASSES),
    )

    # ---- Splits ----
    split_dirs = resolve_split_dirs(dataset_root, yaml_data)

    for s in EXPECTED_SPLITS:
        if s in split_dirs and split_dirs[s].is_dir():
            report.splits_found.append(s)
        else:
            report.splits_missing.append(s)

    logger.info("Splits found: %s | missing: %s", report.splits_found, report.splits_missing)

    # ---- Per-split validation ----
    for split_name in report.splits_found:
        logger.info("Validating split: %s …", split_name)
        split_dir = split_dirs[split_name]
        stats = validate_split(
            split_name,
            split_dir,
            report.num_classes_yaml,
            report.class_names_yaml,
            report.issues,
        )
        report.split_stats[split_name] = stats

    # ---- Pass / fail ----
    critical_types = {
        "missing_file",
        "missing_dir",
        "invalid_names",
        "class_id_out_of_range",
    }
    critical_issues = [i for i in report.issues if i.issue_type in critical_types]
    report.passed = len(critical_issues) == 0 and report.data_yaml_found

    report.elapsed_seconds = round(time.time() - t0, 2)
    return report


# ---------------------------------------------------------------------------
# Terminal report formatting
# ---------------------------------------------------------------------------

_DIVIDER = "─" * 72
_HEADER = "═" * 72


def _resolution_summary(widths: List[int], heights: List[int]) -> str:
    """Format resolution statistics."""
    if not widths:
        return "  No valid images to report."

    lines = [
        f"  Count        : {len(widths)}",
        f"  Width  range : {min(widths)} – {max(widths)} px"
        f"  (mean {statistics.mean(widths):.0f})",
        f"  Height range : {min(heights)} – {max(heights)} px"
        f"  (mean {statistics.mean(heights):.0f})",
    ]
    return "\n".join(lines)


def print_report(report: DatasetReport) -> None:
    """Print a human-readable report to the terminal."""
    print(f"\n{_HEADER}")
    print("  PPE DATASET VALIDATION REPORT")
    print(f"{_HEADER}\n")

    # --- Metadata ---
    print(f"  Dataset path  : {report.dataset_path}")
    print(f"  data.yaml     : {'✔ found' if report.data_yaml_found else '✘ MISSING'}")
    print(f"  Classes (yaml): {report.num_classes_yaml}")
    if report.class_names_yaml:
        for cid in sorted(report.class_names_yaml):
            marker = (
                "✔" if report.class_names_yaml[cid] in REQUIRED_CLASSES else " "
            )
            print(f"    {marker} {cid}: {report.class_names_yaml[cid]}")

    print(f"\n  Required classes found   : {report.required_classes_found}")
    print(f"  Required classes missing : {report.required_classes_missing or 'None'}")

    # --- Splits ---
    print(f"\n{_DIVIDER}")
    print("  SPLIT SUMMARY")
    print(_DIVIDER)

    for split_name in EXPECTED_SPLITS:
        if split_name not in report.split_stats:
            print(f"\n  [{split_name.upper()}]  — not found —")
            continue

        s = report.split_stats[split_name]
        print(f"\n  [{split_name.upper()}]")
        print(f"  Images           : {s.image_count:>7,}")
        print(f"  Labels           : {s.label_count:>7,}")
        print(f"  Annotations      : {s.total_annotations:>7,}")
        print(f"  Missing labels   : {s.images_without_labels:>7,}")
        print(f"  Orphan labels    : {s.labels_without_images:>7,}")
        print(f"  Empty labels     : {s.empty_labels:>7,}")
        print(f"  Corrupt images   : {s.corrupt_images:>7,}")
        print(f"  Invalid bboxes   : {s.invalid_bbox_count:>7,}")
        print(f"  Invalid class IDs: {s.invalid_class_id_count:>7,}")

        if s.class_distribution:
            print("  Class distribution:")
            for cls_name, count in sorted(
                s.class_distribution.items(), key=lambda x: -x[1]
            ):
                max_count = max(s.class_distribution.values(), default=1)
                bar_len = min(count * 40 // max(max_count, 1), 40)
                print(f"    {cls_name:<20s} {count:>7,}  {'█' * bar_len}")

        print("  Resolution:")
        print(_resolution_summary(s.widths, s.heights))

    # --- Issues ---
    print(f"\n{_DIVIDER}")
    print(f"  ISSUES ({len(report.issues)} total)")
    print(_DIVIDER)

    if not report.issues:
        print("  No issues found. ✔")
    else:
        # Group by type
        by_type: Dict[str, List[Issue]] = defaultdict(list)
        for issue in report.issues:
            by_type[issue.issue_type].append(issue)

        for itype, items in sorted(by_type.items()):
            print(f"\n  [{itype}] — {len(items)} occurrence(s)")
            for item in items[:10]:  # Show first 10 per type
                print(f"    {item.split:>6s} | {item.file} | {item.detail}")
            if len(items) > 10:
                print(f"    … and {len(items) - 10} more")

    # --- Verdict ---
    print(f"\n{_HEADER}")
    status = "✔ PASSED" if report.passed else "✘ FAILED (review issues above)"
    print(f"  RESULT: {status}")
    print(f"  Time  : {report.elapsed_seconds:.2f}s")
    print(f"{_HEADER}\n")


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def _report_to_dict(report: DatasetReport) -> Dict[str, Any]:
    """Convert report to a JSON-serialisable dictionary."""
    d: Dict[str, Any] = {
        "dataset_path": report.dataset_path,
        "data_yaml_found": report.data_yaml_found,
        "num_classes_yaml": report.num_classes_yaml,
        "class_names_yaml": {
            str(k): v for k, v in report.class_names_yaml.items()
        },
        "required_classes_found": report.required_classes_found,
        "required_classes_missing": report.required_classes_missing,
        "splits_found": report.splits_found,
        "splits_missing": report.splits_missing,
        "elapsed_seconds": report.elapsed_seconds,
        "passed": report.passed,
        "splits": {},
        "issues": [asdict(i) for i in report.issues],
    }

    for split_name, stats in report.split_stats.items():
        d["splits"][split_name] = {
            "image_count": stats.image_count,
            "label_count": stats.label_count,
            "total_annotations": stats.total_annotations,
            "images_without_labels": stats.images_without_labels,
            "labels_without_images": stats.labels_without_images,
            "empty_labels": stats.empty_labels,
            "corrupt_images": stats.corrupt_images,
            "invalid_bbox_count": stats.invalid_bbox_count,
            "invalid_class_id_count": stats.invalid_class_id_count,
            "class_distribution": stats.class_distribution,
            "resolution": {
                "count": len(stats.widths),
                "width_min": min(stats.widths) if stats.widths else None,
                "width_max": max(stats.widths) if stats.widths else None,
                "width_mean": round(statistics.mean(stats.widths), 1) if stats.widths else None,
                "height_min": min(stats.heights) if stats.heights else None,
                "height_max": max(stats.heights) if stats.heights else None,
                "height_mean": round(statistics.mean(stats.heights), 1) if stats.heights else None,
            },
        }

    return d


def save_json_report(report: DatasetReport, output_path: Path) -> None:
    """Save the validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _report_to_dict(report)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("JSON report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a YOLOv8-format PPE detection dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory (must contain data.yaml).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Also save a JSON report to outputs/dataset_report.json.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Custom path for the JSON report file.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path).resolve()
    if not dataset_root.is_dir():
        logger.error("Dataset path does not exist: %s", dataset_root)
        sys.exit(1)

    report = validate_dataset(dataset_root)
    print_report(report)

    if args.json or args.json_output:
        json_path = (
            Path(args.json_output)
            if args.json_output
            else _PROJECT_ROOT / "outputs" / "dataset_report.json"
        )
        save_json_report(report, json_path)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
