"""
Model Comparison and Accuracy Gain Analysis Script.

Compares the baseline model (trained on the noisy 25-class dataset)
against the retrained model (trained on the cleaned 5-class PPE dataset).

Outputs:
  - Formatted side-by-side terminal comparison table
  - outputs/evaluation/model_comparison.json

Usage:
  python scripts/compare_models.py --baseline models/weights/best.pt --candidate runs/train/ppe_5class_yolov8n/weights/best.pt --data data/processed/ppe_5class/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.evaluate import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("compare_models")


def format_delta(new_val: float, old_val: float, is_percentage: bool = True) -> str:
    """Format difference between new and old values with +/- sign and color indicator."""
    diff = new_val - old_val
    sign = "+" if diff >= 0 else ""
    if is_percentage:
        return f"{sign}{diff * 100:.2f}%"
    return f"{sign}{diff:.4f}"


def compare_models(
    baseline_path: Path,
    candidate_path: Path,
    data_yaml: Path,
    split: str = "test",
    device: str = "auto",
    output_dir: Path = Path("outputs/evaluation"),
) -> Dict[str, Any]:
    """Run evaluation on both models on the same split and compare metrics."""
    if baseline_path.suffix.lower() == ".json" and baseline_path.is_file():
        logger.info("Loading baseline metrics directly from JSON: %s", baseline_path)
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_results = json.load(f)
    else:
        logger.info("Evaluating baseline model: %s", baseline_path)
        base_results, _, _ = evaluate(
            model_path=baseline_path,
            data_yaml=data_yaml,
            split=split,
            device=device,
            output_dir=output_dir / "baseline",
        )

    logger.info("Evaluating candidate 5-class model: %s", candidate_path)
    cand_results, _, _ = evaluate(
        model_path=candidate_path,
        data_yaml=data_yaml,
        split=split,
        device=device,
        output_dir=output_dir / "candidate_5class",
    )

    base_ov = base_results.get("overall", {})
    cand_ov = cand_results.get("overall", {})

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline_model": str(baseline_path),
        "candidate_model": str(candidate_path),
        "dataset_yaml": str(data_yaml),
        "split": split,
        "overall": {
            "mAP50": {
                "baseline": base_ov.get("map50", 0.0),
                "candidate": cand_ov.get("map50", 0.0),
                "delta": round(cand_ov.get("map50", 0.0) - base_ov.get("map50", 0.0), 4),
            },
            "mAP50_95": {
                "baseline": base_ov.get("map50_95", 0.0),
                "candidate": cand_ov.get("map50_95", 0.0),
                "delta": round(cand_ov.get("map50_95", 0.0) - base_ov.get("map50_95", 0.0), 4),
            },
            "precision": {
                "baseline": base_ov.get("precision", 0.0),
                "candidate": cand_ov.get("precision", 0.0),
                "delta": round(cand_ov.get("precision", 0.0) - base_ov.get("precision", 0.0), 4),
            },
            "recall": {
                "baseline": base_ov.get("recall", 0.0),
                "candidate": cand_ov.get("recall", 0.0),
                "delta": round(cand_ov.get("recall", 0.0) - base_ov.get("recall", 0.0), 4),
            },
        },
        "target_classes": {},
    }

    base_tgt = base_results.get("target_classes", {})
    cand_tgt = cand_results.get("target_classes", {})

    all_targets = sorted(set(list(base_tgt.keys()) + list(cand_tgt.keys())))
    for cls_name in all_targets:
        b_item = base_tgt.get(cls_name, {})
        c_item = cand_tgt.get(cls_name, {})

        b_map50 = b_item.get("map50", 0.0)
        c_map50 = c_item.get("map50", 0.0)
        b_rec = b_item.get("recall", 0.0)
        c_rec = c_item.get("recall", 0.0)
        b_prec = b_item.get("precision", 0.0)
        c_prec = c_item.get("precision", 0.0)

        comparison["target_classes"][cls_name] = {
            "instances": c_item.get("instances", b_item.get("instances", 0)),
            "mAP50": {"baseline": b_map50, "candidate": c_map50, "delta": round(c_map50 - b_map50, 4)},
            "recall": {"baseline": b_rec, "candidate": c_rec, "delta": round(c_rec - b_rec, 4)},
            "precision": {"baseline": b_prec, "candidate": c_prec, "delta": round(c_prec - b_prec, 4)},
        }

    # Print Table
    print("\n" + "=" * 84)
    print("  PPE COMPLIANCE DETECTION — MODEL ACCURACY COMPARISON REPORT")
    print("=" * 84)
    print(f"Baseline  : {baseline_path.name}")
    print(f"Candidate : {candidate_path.name} (Optimized 5-Class Architecture)")
    print(f"Split     : {split} ({data_yaml.name})\n")

    print("1. OVERALL METRICS COMPARISON")
    print("-" * 84)
    print(f"{'Metric':<18} | {'Baseline':<12} | {'Candidate (5-Class)':<20} | {'Improvement':<12}")
    print("-" * 84)
    for m_key, m_label in [("mAP50", "mAP@50"), ("mAP50_95", "mAP@50-95"), ("precision", "Precision"), ("recall", "Recall")]:
        b = comparison["overall"][m_key]["baseline"]
        c = comparison["overall"][m_key]["candidate"]
        d_str = format_delta(c, b)
        print(f"{m_label:<18} | {b * 100:>10.2f}% | {c * 100:>18.2f}% | {d_str:>12}")
    print("-" * 84)

    print("\n2. PER-CLASS mAP@50 & RECALL COMPARISON")
    print("-" * 84)
    print(f"{'Class Name':<18} | {'Instances':>9} | {'Base mAP50':>11} | {'New mAP50':>10} | {'Δ mAP50':>10} | {'Δ Recall':>10}")
    print("-" * 84)
    for cls_name, c_data in comparison["target_classes"].items():
        inst = c_data["instances"]
        b_m = c_data["mAP50"]["baseline"]
        c_m = c_data["mAP50"]["candidate"]
        d_m = format_delta(c_m, b_m)
        d_r = format_delta(c_data["recall"]["candidate"], c_data["recall"]["baseline"])
        print(f"{cls_name:<18} | {inst:>9d} | {b_m * 100:>10.2f}% | {c_m * 100:>9.2f}% | {d_m:>10} | {d_r:>10}")
    print("=" * 84 + "\n")

    # Save to file
    out_file = output_dir / "model_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison report saved to: %s", out_file)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs optimized candidate models.")
    parser.add_argument("--baseline", type=str, default="models/weights/best.pt", help="Baseline weights path.")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate 5-class weights path.")
    parser.add_argument("--data", type=str, default="data/processed/ppe_5class/data.yaml", help="Dataset YAML.")
    parser.add_argument("--split", type=str, default="test", help="Evaluation split.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Output directory.")
    args = parser.parse_args()

    compare_models(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        data_yaml=Path(args.data),
        split=args.split,
        device=args.device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
