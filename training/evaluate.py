"""
PPE Model Evaluation Pipeline.

Evaluates trained YOLOv8 models on the Construction Site Safety / PPE dataset,
computing Precision, Recall, mAP@50, and mAP@50-95 both overall and per-class,
with dedicated focus on key PPE compliance classes:
  - Person
  - Hardhat
  - NO-Hardhat
  - Safety Vest
  - NO-Safety Vest

Outputs:
  - outputs/evaluation/results.json  (Structured JSON report)
  - outputs/evaluation/report.txt    (Human-readable formatted report)

Usage:
    python training/evaluate.py --model models/weights/best.pt --data path/to/dataset --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_logger
from training.train import parse_data_yaml, prepare_data_yaml

logger = get_logger("evaluate")

# Required target classes for PPE compliance
TARGET_PPE_CLASSES: List[str] = [
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
]


# ---------------------------------------------------------------------------
# Path & argument validation
# ---------------------------------------------------------------------------


def validate_model_path(model_path: Path) -> Path:
    """Validate that the model weights file exists.

    Args:
        model_path: Path to the .pt file.

    Returns:
        The validated Path object.

    Raises:
        SystemExit: If the file does not exist or is not a file.
    """
    if not model_path.exists():
        logger.error(
            "Model weights file not found: %s\n"
            "Please provide a valid path using --model (e.g. models/weights/best.pt).",
            model_path.resolve(),
        )
        sys.exit(1)
    if not model_path.is_file():
        logger.error("Specified model path is not a file: %s", model_path.resolve())
        sys.exit(1)
    return model_path


def validate_data_path(data_path: Path) -> Path:
    """Validate that data.yaml exists or locate it within a dataset directory.

    Args:
        data_path: Path to data.yaml or dataset directory.

    Returns:
        Path to the valid data.yaml.

    Raises:
        SystemExit: If data.yaml cannot be located.
    """
    if data_path.is_file():
        return data_path

    if data_path.is_dir():
        cand = data_path / "data.yaml"
        if cand.is_file():
            return cand
        cand_yml = data_path / "data.yml"
        if cand_yml.is_file():
            return cand_yml

    logger.error(
        "Dataset configuration 'data.yaml' not found in: %s\n"
        "Please provide a valid path via --data.",
        data_path.resolve(),
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_evaluation_metrics(
    metrics: Any,
    model_names: Dict[int, str],
    target_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract metrics from an Ultralytics DetMetrics object into a clean dictionary.

    Args:
        metrics: DetMetrics object returned by model.val().
        model_names: Mapping of class index to class name.
        target_classes: Optional list of class names to highlight.

    Returns:
        Structured dictionary of overall, target, and per-class metrics.
    """
    if target_classes is None:
        target_classes = TARGET_PPE_CLASSES

    # Overall metrics
    overall_p = float(metrics.box.mp)
    overall_r = float(metrics.box.mr)
    overall_map50 = float(metrics.box.map50)
    overall_map50_95 = float(metrics.box.map)

    overall_metrics = {
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "map50": round(overall_map50, 4),
        "map50_95": round(overall_map50_95, 4),
    }

    # Speed metrics
    speed_dict = getattr(metrics, "speed", {}) or {}
    preprocess_ms = float(speed_dict.get("preprocess", 0.0))
    inference_ms = float(speed_dict.get("inference", 0.0))
    loss_ms = float(speed_dict.get("loss", 0.0))
    postprocess_ms = float(speed_dict.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0

    speed_metrics = {
        "preprocess_ms": round(preprocess_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "loss_ms": round(loss_ms, 2),
        "postprocess_ms": round(postprocess_ms, 2),
        "total_latency_ms": round(total_ms, 2),
        "fps": fps,
    }

    # Per-class metrics
    # ap_class_index maps array index i to model class index
    ap_indices = getattr(metrics.box, "ap_class_index", None)
    if ap_indices is None or len(ap_indices) == 0:
        ap_indices = np.array(list(range(len(metrics.box.p))))

    nt_per_class = getattr(metrics, "nt_per_class", None)

    per_class_results: Dict[str, Dict[str, Any]] = {}
    target_results: Dict[str, Dict[str, Any]] = {}

    for i, cls_idx_val in enumerate(ap_indices):
        cls_idx = int(cls_idx_val)
        cls_name = model_names.get(cls_idx, f"class_{cls_idx}")

        # Metrics for class i
        p = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0
        r = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0
        ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
        ap = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0

        instances = 0
        if nt_per_class is not None and cls_idx < len(nt_per_class):
            instances = int(nt_per_class[cls_idx])

        entry = {
            "class_id": cls_idx,
            "class_name": cls_name,
            "instances": instances,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(ap50, 4),
            "map50_95": round(ap, 4),
        }
        per_class_results[cls_name] = entry

    # Fill target classes
    for target in target_classes:
        if target in per_class_results:
            target_results[target] = per_class_results[target]
        else:
            # Target class present in model names but had 0 instances in this split
            matched_id = None
            for cid, cname in model_names.items():
                if cname == target:
                    matched_id = cid
                    break
            target_results[target] = {
                "class_id": matched_id if matched_id is not None else -1,
                "class_name": target,
                "instances": 0,
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }

    return {
        "overall": overall_metrics,
        "speed": speed_metrics,
        "target_classes": target_results,
        "per_class": per_class_results,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_text_report(data: Dict[str, Any]) -> str:
    """Generate a clean, professional academic evaluation report.

    Args:
        data: Evaluation results dictionary.

    Returns:
        Formatted multi-line report string.
    """
    lines: List[str] = []
    sep_heavy = "=" * 76
    sep_light = "-" * 76

    lines.append(sep_heavy)
    lines.append("  REAL-TIME PPE COMPLIANCE DETECTION — MODEL EVALUATION REPORT")
    lines.append(sep_heavy)
    lines.append(f"  Timestamp    : {data.get('timestamp', 'N/A')}")
    lines.append(f"  Model Path   : {data.get('model_path', 'N/A')}")
    lines.append(f"  Dataset YAML : {data.get('data_yaml', 'N/A')}")
    lines.append(f"  Split        : {data.get('split', 'N/A')}")
    lines.append(f"  Device       : {data.get('device', 'N/A')}")
    lines.append(sep_light)

    # 1. Overall Summary
    overall = data.get("overall", {})
    lines.append("\n1. OVERALL EVALUATION METRICS (All Classes)")
    lines.append(sep_light)
    lines.append(
        f"  {'Precision':<16s}: {overall.get('precision', 0.0):.4f} "
        f"({overall.get('precision', 0.0) * 100:.2f}%)"
    )
    lines.append(
        f"  {'Recall':<16s}: {overall.get('recall', 0.0):.4f} "
        f"({overall.get('recall', 0.0) * 100:.2f}%)"
    )
    lines.append(
        f"  {'mAP@50':<16s}: {overall.get('map50', 0.0):.4f} "
        f"({overall.get('map50', 0.0) * 100:.2f}%)"
    )
    lines.append(
        f"  {'mAP@50-95':<16s}: {overall.get('map50_95', 0.0):.4f} "
        f"({overall.get('map50_95', 0.0) * 100:.2f}%)"
    )

    # 2. Target PPE Classes Table
    lines.append("\n2. TARGET PPE COMPLIANCE CLASSES")
    lines.append(sep_light)
    header = (
        f"  {'Class Name':<18s} | {'Instances':>9s} | "
        f"{'Precision':>9s} | {'Recall':>8s} | {'mAP@50':>8s} | {'mAP@50-95':>9s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 72)

    target_classes = data.get("target_classes", {})
    for _, item in target_classes.items():
        name = item.get("class_name", "Unknown")
        inst = item.get("instances", 0)
        p = item.get("precision", 0.0)
        r = item.get("recall", 0.0)
        map50 = item.get("map50", 0.0)
        map50_95 = item.get("map50_95", 0.0)
        row = (
            f"  {name:<18s} | {inst:>9d} | "
            f"{p:>9.4f} | {r:>8.4f} | {map50:>8.4f} | {map50_95:>9.4f}"
        )
        lines.append(row)

    # 3. Full Per-Class Breakdown
    lines.append("\n3. ALL CLASSES BREAKDOWN")
    lines.append(sep_light)
    lines.append(header)
    lines.append("  " + "-" * 72)

    per_class = data.get("per_class", {})
    for _, item in sorted(
        per_class.items(), key=lambda kv: kv[1].get("map50", 0.0), reverse=True
    ):
        name = item.get("class_name", "Unknown")
        inst = item.get("instances", 0)
        p = item.get("precision", 0.0)
        r = item.get("recall", 0.0)
        map50 = item.get("map50", 0.0)
        map50_95 = item.get("map50_95", 0.0)
        row = (
            f"  {name:<18s} | {inst:>9d} | "
            f"{p:>9.4f} | {r:>8.4f} | {map50:>8.4f} | {map50_95:>9.4f}"
        )
        lines.append(row)

    # 4. Latency & Throughput Benchmark
    speed = data.get("speed", {})
    lines.append("\n4. SPEED & LATENCY BENCHMARK (Real-Time Performance)")
    lines.append(sep_light)
    lines.append(f"  Preprocess Latency  : {speed.get('preprocess_ms', 0.0):.2f} ms")
    lines.append(f"  Inference Latency   : {speed.get('inference_ms', 0.0):.2f} ms")
    lines.append(f"  Postprocess Latency : {speed.get('postprocess_ms', 0.0):.2f} ms")
    lines.append(f"  Total Latency/Frame : {speed.get('total_latency_ms', 0.0):.2f} ms")
    lines.append(f"  Throughput (FPS)    : {speed.get('fps', 0.0):.1f} FPS")
    lines.append(sep_heavy)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Evaluation Routine
# ---------------------------------------------------------------------------


def evaluate(
    model_path: Path,
    data_yaml: Path,
    split: str = "val",
    device: str = "auto",
    output_dir: Path = Path("outputs/evaluation"),
) -> Tuple[Dict[str, Any], Path, Path]:
    """Run evaluation and save both JSON and human-readable text reports.

    Args:
        model_path: Path to trained model weights (.pt).
        data_yaml: Path to data.yaml.
        split: Split name ('val', 'test', 'train').
        device: Device string ('auto', 'cpu', '0').
        output_dir: Output directory for reports.

    Returns:
        Tuple of (results_dict, json_path, txt_path).
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

    # Prepare data.yaml in output_dir if path adjustments needed
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_yaml = prepare_data_yaml(data_yaml, output_dir)

    # Load model
    logger.info("Loading model weights from: %s", model_path)
    model = YOLO(str(model_path))

    # Run evaluation
    logger.info(
        "Evaluating model on split='%s' using device='%s' …",
        split,
        resolved_device,
    )
    t0 = time.time()
    metrics = model.val(
        data=str(fixed_yaml),
        split=split,
        device=resolved_device,
        verbose=False,
    )
    eval_duration = time.time() - t0
    logger.info("Evaluation completed in %.2fs", eval_duration)

    # Extract structured results
    model_names = model.names if isinstance(model.names, dict) else {}
    results = extract_evaluation_metrics(metrics, model_names, TARGET_PPE_CLASSES)

    # Add metadata
    results["timestamp"] = datetime.now().isoformat()
    results["model_path"] = str(model_path.resolve())
    results["data_yaml"] = str(data_yaml.resolve())
    results["split"] = split
    results["device"] = resolved_device
    results["evaluation_time_seconds"] = round(eval_duration, 2)

    # Save outputs/evaluation/results.json
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to: %s", json_path)

    # Save outputs/evaluation/report.txt
    txt_path = output_dir / "report.txt"
    report_text = generate_text_report(results)
    txt_path.write_text(report_text, encoding="utf-8")
    logger.info("Human-readable report saved to: %s", txt_path)

    # Print report to stdout
    print("\n" + report_text + "\n")

    return results, json_path, txt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLOv8n model for PPE compliance detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/weights/best.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml or dataset root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for evaluation: 'auto', 'cpu', '0', etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory where results.json and report.txt are saved.",
    )
    return parser


def main() -> None:
    """Main entrypoint for evaluate.py."""
    parser = build_parser()
    args = parser.parse_args()

    model_path = validate_model_path(Path(args.model))
    data_yaml = validate_data_path(Path(args.data))
    output_dir = Path(args.output_dir)

    evaluate(
        model_path=model_path,
        data_yaml=data_yaml,
        split=args.split,
        device=args.device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
