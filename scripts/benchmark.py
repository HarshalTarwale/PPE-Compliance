"""
Comprehensive Pipeline Benchmarking Script.

Measures actual, empirical performance metrics across each isolated stage and the full pipeline:
  1. YOLOv8 Detection Latency & FPS
  2. Spatial Person-to-PPE Association Latency
  3. Face Detection & Feature Extraction Latency (YuNet + SFace)
  4. Temporal Smoothing & Voting Latency
  5. End-to-End Pipeline Throughput (P50, P95, P99, Mean FPS)
  6. Memory Footprint (VRAM / RAM)

Outputs:
  - Terminal formatted benchmark table
  - outputs/benchmark_results.json

Usage:
  python scripts/benchmark.py --iterations 100 --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compliance.rules import ComplianceEngine
from compliance.temporal_filter import PPETemporalFilter
from detection.detector import Detection, PPEDetector
from detection.pipeline import PPECompliancePipeline
from detection.ppe_association import PPEAssociator
from recognition.embeddings import FaceEmbeddingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("benchmark")


def create_synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a realistic test image with geometric patterns to ensure non-trivial inference."""
    np.random.seed(42)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add gradient and shapes
    for y in range(height):
        img[y, :, :] = (y % 120, (y * 2) % 180, (y * 3) % 255)
    # Draw simulated worker silhouettes
    cv2.rectangle(img, (150, 80), (320, 440), (120, 180, 220), -1)
    cv2.rectangle(img, (180, 85), (290, 150), (0, 200, 0), -1)   # Hardhat
    cv2.rectangle(img, (160, 180), (310, 360), (50, 205, 50), -1) # Vest
    return img


def benchmark_pipeline(
    iterations: int = 50,
    device: str = "auto",
    model_path: str = "models/weights/best.pt",
    output_json: str = "outputs/benchmark_results.json",
    warmup: int = 10,
    half: bool = False,
) -> Dict[str, Any]:
    """Run empirical benchmarks across isolated components and the unified pipeline."""
    logger.info("Initializing components for benchmark (Device: %s, FP16: %s)...", device, half)

    frame = create_synthetic_frame()

    # 1. Initialize Components
    detector = PPEDetector(model_path=model_path, device=device, half=half)
    associator = PPEAssociator()
    temporal_filter = PPETemporalFilter()
    compliance_engine = ComplianceEngine()
    face_engine = FaceEmbeddingEngine()
    pipeline = PPECompliancePipeline(
        detector=detector,
        associator=associator,
        temporal_filter=temporal_filter,
        compliance_engine=compliance_engine,
    )

    actual_device = str(detector.device)
    logger.info("Detector active on device: %s", actual_device)

    # -----------------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------------
    logger.info("Running %d warmup iterations...", warmup)
    for _ in range(warmup):
        _ = detector.detect(frame)
        _ = pipeline.process_frame(frame, draw_overlay=False)

    if torch.cuda.is_available() and "cuda" in actual_device:
        torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Benchmark 1: Isolated YOLO Detection
    # -----------------------------------------------------------------------
    logger.info("Benchmarking YOLOv8 Detection (%d iterations)...", iterations)
    det_latencies: List[float] = []

    for _ in range(iterations):
        if torch.cuda.is_available() and "cuda" in actual_device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        det_res = detector.detect(frame)
        if torch.cuda.is_available() and "cuda" in actual_device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        det_latencies.append((t1 - t0) * 1000.0)

    # -----------------------------------------------------------------------
    # Benchmark 2: Isolated Spatial Association
    # -----------------------------------------------------------------------
    logger.info("Benchmarking Spatial PPE Association (%d iterations)...", iterations)
    sample_persons = [Detection(150, 80, 320, 440, class_id=8, class_name="Person", confidence=0.92)]
    sample_ppe = [
        Detection(180, 85, 290, 150, class_id=2, class_name="Hardhat", confidence=0.88),
        Detection(160, 180, 310, 360, class_id=11, class_name="Safety Vest", confidence=0.85),
    ]

    assoc_latencies: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = associator.associate(sample_persons, sample_ppe)
        t1 = time.perf_counter()
        assoc_latencies.append((t1 - t0) * 1000.0)

    # -----------------------------------------------------------------------
    # Benchmark 3: Isolated Temporal Smoothing
    # -----------------------------------------------------------------------
    logger.info("Benchmarking Temporal Filter (%d iterations)...", iterations)
    from compliance.rules import PersonPPEInfo

    filter_latencies: List[float] = []
    info = PersonPPEInfo("bench_person", has_hardhat=True, has_safety_vest=True)
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = temporal_filter.update(info)
        t1 = time.perf_counter()
        filter_latencies.append((t1 - t0) * 1000.0)

    # -----------------------------------------------------------------------
    # Benchmark 4: Isolated Face Feature Extraction (SFace)
    # -----------------------------------------------------------------------
    logger.info("Benchmarking Face Alignment & Embedding (%d iterations)...", iterations)
    face_latencies: List[float] = []
    head_crop = frame[80:200, 150:320]

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = face_engine.extract_single_face_embedding(head_crop)
        t1 = time.perf_counter()
        face_latencies.append((t1 - t0) * 1000.0)

    # -----------------------------------------------------------------------
    # Benchmark 5: End-to-End Pipeline
    # -----------------------------------------------------------------------
    logger.info("Benchmarking Full End-to-End Pipeline (%d iterations)...", iterations)
    e2e_latencies: List[float] = []

    for _ in range(iterations):
        if torch.cuda.is_available() and "cuda" in actual_device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipeline.process_frame(frame, draw_overlay=True)
        if torch.cuda.is_available() and "cuda" in actual_device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_latencies.append((t1 - t0) * 1000.0)

    # -----------------------------------------------------------------------
    # Memory Statistics
    # -----------------------------------------------------------------------
    vram_mb = 0.0
    if torch.cuda.is_available() and "cuda" in actual_device:
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # -----------------------------------------------------------------------
    # Aggregate Statistics
    # -----------------------------------------------------------------------
    def calc_stats(lat_list: List[float]) -> Dict[str, float]:
        arr = np.array(lat_list)
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "fps": round(1000.0 / float(np.mean(arr)), 1) if np.mean(arr) > 0 else 0.0,
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "device": actual_device,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
            "vram_peak_mb": round(vram_mb, 1),
        },
        "configuration": {
            "iterations": iterations,
            "image_size": [frame.shape[1], frame.shape[0]],
            "model_path": str(model_path),
        },
        "benchmarks": {
            "yolo_detection": calc_stats(det_latencies),
            "spatial_association": calc_stats(assoc_latencies),
            "face_embedding": calc_stats(face_latencies),
            "temporal_smoothing": calc_stats(filter_latencies),
            "end_to_end_pipeline": calc_stats(e2e_latencies),
        },
    }

    # Save to JSON
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 76)
    print("  PPE COMPLIANCE SYSTEM — EMPIRICAL BENCHMARK REPORT")
    print("=" * 76)
    print(f"Device: {results['hardware']['device_name']} ({actual_device}) | VRAM: {vram_mb:.1f} MB")
    print(f"Test Iterations: {iterations} frames @ 640x480 resolution\n")

    header = f"{'Pipeline Stage':<28} | {'Mean (ms)':<10} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | {'FPS':<8}"
    print(header)
    print("-" * 76)

    for stage_name, stats in results["benchmarks"].items():
        formatted_name = stage_name.replace("_", " ").title()
        print(
            f"{formatted_name:<28} | {stats['mean_ms']:<10.2f} | {stats['p95_ms']:<9.2f} | {stats['p99_ms']:<9.2f} | {stats['fps']:<8.1f}"
        )

    print("=" * 76)
    print(f"Results saved to: {out_path.resolve()}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PPE compliance pipeline.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', '0', 'cpu').")
    parser.add_argument("--model", type=str, default="models/weights/best.pt", help="Path to weights.")
    parser.add_argument("--output", type=str, default="outputs/benchmark_results.json", help="Output JSON path.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 half-precision inference on GPU.")
    args = parser.parse_args()

    benchmark_pipeline(
        iterations=args.iterations,
        device=args.device,
        model_path=args.model,
        output_json=args.output,
        half=args.half,
    )


if __name__ == "__main__":
    main()
