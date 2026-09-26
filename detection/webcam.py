"""
Real-Time PPE Detection using Webcam.

Captures live video feed from a webcam, runs YOLOv8 inference frame-by-frame,
and renders bounding boxes, class names, confidence scores, and real-time
telemetry (FPS and latency).

Usage:
    # Default webcam (index 0)
    python detection/webcam.py

    # Specific camera index or resolution
    python detection/webcam.py --camera 1 --width 1280 --height 720

    # Custom confidence and model weights
    python detection/webcam.py --model models/weights/best.pt --confidence 0.35
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Union

import cv2

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_logger
from detection.detector import PPEDetector

logger = get_logger("webcam")


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def parse_camera_source(source: str) -> Union[int, str]:
    """Parse camera argument into an integer index or string path.

    Args:
        source: String such as "0", "1", or a video filepath.

    Returns:
        Integer device index if digits, otherwise original string.
    """
    cleaned = source.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return cleaned


def open_camera(
    source: Union[int, str],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[cv2.VideoCapture]:
    """Initialize OpenCV video capture with optional resolution.

    Args:
        source: Camera index (int) or video source (str).
        width: Optional requested frame width.
        height: Optional requested frame height.

    Returns:
        Open VideoCapture object, or None if initialization failed.
    """
    logger.info("Opening camera source: %s", source)

    # Use CAP_DSHOW on Windows for faster initialization if int index
    if sys.platform == "win32" and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(
            "Failed to initialize camera source '%s'.\n"
            "Troubleshooting tips:\n"
            "  1. Ensure your webcam is connected and not used by another application.\n"
            "  2. Check Windows Settings -> Privacy & security -> Camera permissions.\n"
            "  3. Try another camera index (e.g. --camera 1).",
            source,
        )
        return None

    # Set frame dimensions if requested
    if width is not None and width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera initialized: %dx%d @ %.1f FPS", actual_w, actual_h, cap.get(cv2.CAP_PROP_FPS))

    return cap


# ---------------------------------------------------------------------------
# Webcam Detection Loop
# ---------------------------------------------------------------------------


def run_webcam(
    camera_source: Union[int, str] = 0,
    model_path: Union[str, Path] = "models/weights/best.pt",
    confidence: float = 0.25,
    width: Optional[int] = None,
    height: Optional[int] = None,
    device: str = "auto",
    headless: bool = False,
    max_frames: Optional[int] = None,
) -> int:
    """Run real-time PPE detection on webcam stream.

    Args:
        camera_source: Camera index (int) or video path (str).
        model_path: Path to YOLO model weights.
        confidence: Minimum confidence threshold.
        width: Optional frame width.
        height: Optional frame height.
        device: 'auto', 'cpu', '0', etc.
        headless: If True, disables cv2.imshow (for headless testing).
        max_frames: Optional limit on total frames to process.

    Returns:
        Total number of frames processed.

    Raises:
        SystemExit: If camera initialization or model loading fails.
    """
    # 1. Initialize detector
    try:
        detector = PPEDetector(
            model_path=model_path,
            confidence=confidence,
            device=device,
        )
    except FileNotFoundError as err:
        logger.error("%s", err)
        sys.exit(1)
    except Exception as err:
        logger.error("Failed to load model: %s", err)
        sys.exit(1)

    # 2. Open camera
    cap = open_camera(camera_source, width=width, height=height)
    if cap is None:
        logger.error("Exiting due to camera initialization failure.")
        sys.exit(1)

    window_name = "Real-Time PPE Detection (YOLOv8n) — Press 'Q' to Exit"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Rolling FPS window (last 30 frames)
    fps_history = deque(maxlen=30)
    total_frames = 0
    total_latency_accum = 0.0

    logger.info("Starting live detection loop. Press 'Q' or 'Esc' to exit.")

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Empty or failed frame read. Ending video stream.")
                break

            total_frames += 1

            # Run inference
            result = detector.detect(frame, confidence=confidence)
            total_latency_accum += result.inference_ms

            # Compute smoothed FPS
            frame_dt = time.perf_counter() - t_frame_start
            if frame_dt > 0:
                fps_history.append(1.0 / frame_dt)
            current_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            # Render visual overlays
            detector.draw_detections(frame, result.detections, copy=False)
            detector.draw_hud(
                frame,
                fps=current_fps,
                inference_ms=result.inference_ms,
                detection_count=result.count,
            )

            # Display frame
            if not headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q', 'Q', or Esc
                    logger.info("Quit command received ('%s'). Exiting.", chr(key) if key != 27 else "Esc")
                    break

            if max_frames is not None and total_frames >= max_frames:
                logger.info("Reached maximum frame limit (%d). Exiting.", max_frames)
                break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    avg_fps = (total_frames / (sum(1.0 / f for f in fps_history) if fps_history else 1.0)) if total_frames > 0 else 0.0
    avg_latency = (total_latency_accum / total_frames) if total_frames > 0 else 0.0

    logger.info(
        "Session complete: %d frames processed | Avg Latency: %.2f ms | Avg FPS: %.1f",
        total_frames,
        avg_latency,
        avg_fps,
    )
    return total_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Real-Time PPE Compliance Detection using Webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/weights/best.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera index (e.g. 0, 1) or video source path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (0.0 to 1.0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional webcam capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional webcam capture height in pixels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'auto', '0', 'cpu', etc.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window (for automated testing or servers).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to process before exiting.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for webcam detection."""
    parser = build_parser()
    args = parser.parse_args()

    camera_source = parse_camera_source(args.camera)

    run_webcam(
        camera_source=camera_source,
        model_path=args.model,
        confidence=args.confidence,
        width=args.width,
        height=args.height,
        device=args.device,
        headless=args.headless,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
