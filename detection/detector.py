"""
PPE Detection Inference Engine.

Provides the core YOLOv8 inference wrapper, abstracting model execution,
bounding box parsing, latency measurement, and visualization rendering.

Separates detection/inference logic from any UI/camera capture code.
"""

from __future__ import annotations

import ast
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_logger

logger = get_logger("detector")

# Curated palette for PPE classes in BGR format
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Person": (230, 160, 40),       # Blue-ish / Teal
    "Hardhat": (0, 200, 0),         # Vibrant Green (Compliant)
    "NO-Hardhat": (0, 0, 230),       # Bright Red (Violation)
    "Safety Vest": (50, 205, 50),   # Neon / Lime Green (Compliant)
    "NO-Safety Vest": (0, 100, 255), # Orange-Red (Violation)
    "Mask": (200, 200, 0),          # Cyan
    "NO-Mask": (0, 140, 255),       # Orange
    "Gloves": (180, 105, 255),      # Pink/Purple
    "Safety Cone": (0, 165, 255),   # Orange
}

DEFAULT_BOX_COLOR: Tuple[int, int, int] = (180, 180, 180)  # Light gray for other classes


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """Represents a single detected object."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2)."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


@dataclass
class DetectionResult:
    """Contains all detections and timing telemetry for a single frame."""

    detections: List[Detection] = field(default_factory=list)
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0
    orig_shape: Tuple[int, int] = (0, 0)  # (height, width)

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_class(self, class_names: List[str]) -> List[Detection]:
        """Return detections belonging to any of the specified class names."""
        target_set = set(class_names)
        return [d for d in self.detections if d.class_name in target_set]


# ---------------------------------------------------------------------------
# PPEDetector Engine
# ---------------------------------------------------------------------------


DEFAULT_ONNX_MODEL = "models/weights/best_5class.onnx"


def resolve_detector_backend() -> str:
    """Pick 'onnx' or 'torch' for the detector.

    PPE_DETECTOR_BACKEND overrides. Otherwise, use the lightweight ONNX Runtime
    backend on Render (free instances have 512 MB RAM, and PyTorch alone needs
    more than that) or when PyTorch/Ultralytics are not installed.
    """
    env = os.getenv("PPE_DETECTOR_BACKEND", "").strip().lower()
    if env in ("onnx", "torch"):
        return env
    if os.getenv("RENDER"):
        return "onnx"
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None or importlib.util.find_spec("torch") is None:
        return "onnx"
    return "torch"


class PPEDetector:
    """High-performance YOLOv8 PPE detector."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/weights/best.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
        half: Optional[bool] = None,
    ) -> None:
        """Initialize the PPE detector.

        Args:
            model_path: Path to YOLO weights (.pt).
            confidence: Default confidence threshold (0.0 to 1.0).
            iou: NMS IoU threshold (0.0 to 1.0).
            device: 'auto', 'cpu', '0', etc.
            imgsz: Inference image size.
            half: Whether to use FP16 half-precision inference (auto-enabled on CUDA).

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            # If relative, check from project root
            cand = _PROJECT_ROOT / self.model_path
            if cand.is_file():
                self.model_path = cand
            else:
                raise FileNotFoundError(
                    f"Model weights not found at: {self.model_path.resolve()}"
                )

        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.backend = "onnx" if self.model_path.suffix.lower() == ".onnx" else "torch"

        if self.backend == "onnx":
            self._init_onnx()
            return

        # Resolve device
        if device == "auto":
            import torch  # type: ignore

            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # FP16 is only valid on CUDA devices (auto-enable by default on GPU)
        if half is None:
            self.half = (self.device != "cpu")
        else:
            self.half = bool(half) and (self.device != "cpu")

        logger.info(
            "Loading YOLO detector: weights='%s', device='%s', conf=%.2f, iou=%.2f, half=%s",
            self.model_path.name,
            self.device,
            self.confidence,
            self.iou,
            self.half,
        )

        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(str(self.model_path))

        # Convert weights to FP16 half precision if requested on CUDA
        if self.half and hasattr(self.model, "model") and self.model.model is not None:
            try:
                target_dev = f"cuda:{self.device}" if self.device.isdigit() else self.device
                self.model.to(target_dev)
                self.model.model.half()
            except Exception as e:
                logger.warning("Could not convert model to FP16: %s", e)
                self.half = False

        # Cache class names mapping
        self.names: Dict[int, str] = {}
        if isinstance(self.model.names, dict):
            self.names = {int(k): str(v) for k, v in self.model.names.items()}
        elif isinstance(self.model.names, list):
            self.names = {i: str(n) for i, n in enumerate(self.model.names)}

        logger.info("Detector ready with %d classes", len(self.names))

    def detect(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
    ) -> DetectionResult:
        """Run object detection on a BGR image frame.

        Args:
            frame: Input image as numpy array in BGR format.
            confidence: Optional confidence threshold override.

        Returns:
            DetectionResult containing detections and timing info.
        """
        if frame is None or frame.size == 0:
            return DetectionResult()

        h, w = frame.shape[:2]
        conf = confidence if confidence is not None else self.confidence

        if self.backend == "onnx":
            return self._detect_onnx(frame, conf)

        t0 = time.perf_counter()

        # Run Ultralytics inference
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        wall_latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        speed_dict: Dict[str, float] = {}

        if results and len(results) > 0:
            res = results[0]
            speed_dict = getattr(res, "speed", {}) or {}

            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    x1 = max(0, int(round(xyxy[i][0])))
                    y1 = max(0, int(round(xyxy[i][1])))
                    x2 = min(w, int(round(xyxy[i][2])))
                    y2 = min(h, int(round(xyxy[i][3])))
                    c_id = int(clss[i])
                    c_name = self.names.get(c_id, f"class_{c_id}")
                    score = float(confs[i])

                    detections.append(
                        Detection(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            class_id=c_id,
                            class_name=c_name,
                            confidence=score,
                        )
                    )

        # Extract timing
        prep = float(speed_dict.get("preprocess", 0.0))
        infer = float(speed_dict.get("inference", 0.0))
        post = float(speed_dict.get("postprocess", 0.0))
        total = prep + infer + post if (prep + infer + post) > 0 else wall_latency_ms
        fps = round(1000.0 / total, 1) if total > 0 else 0.0

        return DetectionResult(
            detections=detections,
            preprocess_ms=round(prep, 2),
            inference_ms=round(infer, 2),
            postprocess_ms=round(post, 2),
            total_latency_ms=round(total, 2),
            fps=fps,
            orig_shape=(h, w),
        )

    # -----------------------------------------------------------------------
    # ONNX Runtime backend (no PyTorch needed; low memory for cloud hosting)
    # -----------------------------------------------------------------------

    def _init_onnx(self) -> None:
        import onnxruntime as ort  # type: ignore

        opts = ort.SessionOptions()
        # Render free instances get a fraction of one CPU; extra threads only add overhead
        threads = int(os.getenv("ONNX_THREADS") or (1 if os.getenv("RENDER") else 0))
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.device = "cpu"
        self.half = False
        self.model = None
        self._input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        if isinstance(in_shape[2], int):
            self.imgsz = int(in_shape[2])

        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = {}
        if "names" in meta:
            try:
                self.names = {int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()}
            except (ValueError, SyntaxError):
                self.names = {}
        logger.info(
            "Loaded ONNX detector '%s' (onnxruntime, cpu) with %d classes",
            self.model_path.name,
            len(self.names),
        )

    def _detect_onnx(self, frame: np.ndarray, conf: float) -> DetectionResult:
        h, w = frame.shape[:2]
        size = self.imgsz

        # Letterbox resize (same as Ultralytics: keep aspect ratio, pad with 114)
        t0 = time.perf_counter()
        r = min(size / h, size / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        pad_x, pad_y = (size - new_w) / 2.0, (size - new_h) / 2.0
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, size - new_h - top, left, size - new_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, swapRB=True)
        t1 = time.perf_counter()

        out = self.session.run(None, {self._input_name: blob})[0]
        t2 = time.perf_counter()

        # Output: [1, 4 + num_classes, N] -> [N, 4 + num_classes]
        preds = out[0].T
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]
        keep = scores >= conf
        preds, class_ids, scores = preds[keep], class_ids[keep], scores[keep]

        detections: List[Detection] = []
        if len(scores) > 0:
            cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            x1 = (cx - bw / 2 - left) / r
            y1 = (cy - bh / 2 - top) / r
            x2 = (cx + bw / 2 - left) / r
            y2 = (cy + bh / 2 - top) / r

            # Class-aware NMS: shift each class far apart so classes never suppress each other
            offset = class_ids.astype(np.float32) * 4096.0
            nms_boxes = np.stack([x1 + offset, y1, x2 - x1, y2 - y1], axis=1).tolist()
            idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.astype(float).tolist(), conf, self.iou)
            idxs = np.array(idxs, dtype=int).reshape(-1)[:300]

            for i in idxs:
                c_id = int(class_ids[i])
                detections.append(
                    Detection(
                        x1=max(0, int(round(x1[i]))),
                        y1=max(0, int(round(y1[i]))),
                        x2=min(w, int(round(x2[i]))),
                        y2=min(h, int(round(y2[i]))),
                        class_id=c_id,
                        class_name=self.names.get(c_id, f"class_{c_id}"),
                        confidence=float(scores[i]),
                    )
                )
        t3 = time.perf_counter()

        prep, infer, post = (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000
        total = prep + infer + post
        return DetectionResult(
            detections=detections,
            preprocess_ms=round(prep, 2),
            inference_ms=round(infer, 2),
            postprocess_ms=round(post, 2),
            total_latency_ms=round(total, 2),
            fps=round(1000.0 / total, 1) if total > 0 else 0.0,
            orig_shape=(h, w),
        )

    # -----------------------------------------------------------------------
    # Rendering Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def get_class_color(class_name: str) -> Tuple[int, int, int]:
        """Return the BGR color for a given class name."""
        if class_name in CLASS_COLORS_BGR:
            return CLASS_COLORS_BGR[class_name]
        # Hash to consistent color for unknown classes
        val = abs(hash(class_name)) % 360
        # Simple HSV to BGR
        h_val = np.uint8(val // 2)
        hsv_pixel = np.uint8([[[h_val, 200, 220]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2,
        font_scale: float = 0.5,
        copy: bool = False,
    ) -> np.ndarray:
        """Render bounding boxes and class badges onto an image.

        Args:
            frame: Input BGR image.
            detections: List of Detection objects.
            line_thickness: Box stroke thickness.
            font_scale: Text font scale.
            copy: Whether to create a copy of the frame.

        Returns:
            Annotated BGR frame.
        """
        out = frame.copy() if copy else frame

        for det in detections:
            color = self.get_class_color(det.class_name)

            # Draw bounding box
            cv2.rectangle(
                out,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )

            # Format label
            label = f"{det.class_name} {det.confidence:.0%}"

            # Calculate text size for badge background
            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                1,
            )

            # Badge coordinates (place above box if fits, else inside)
            bx1 = det.x1
            by1 = max(0, det.y1 - th - baseline - 4)
            bx2 = det.x1 + tw + 6
            by2 = det.y1

            if by1 < 0:
                by1 = det.y1
                by2 = det.y1 + th + baseline + 4

            # Draw filled pill background
            cv2.rectangle(out, (bx1, by1), (bx2, by2), color, cv2.FILLED)

            # Text color (black for light colors, white for dark)
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            txt_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)

            # Draw text
            cv2.putText(
                out,
                label,
                (bx1 + 3, by2 - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                txt_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return out

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        inference_ms: float,
        detection_count: int,
        model_name: str = "YOLOv8n PPE",
    ) -> np.ndarray:
        """Render a clean, semi-transparent Heads-Up Display (HUD) overlay.

        Args:
            frame: Input BGR image.
            fps: Current frame rate.
            inference_ms: Model inference latency in milliseconds.
            detection_count: Total detections in current frame.
            model_name: Model title.

        Returns:
            Frame with HUD overlay.
        """
        hud_w, hud_h = 240, 95
        x, y = 12, 12

        # Draw semi-transparent dark background card
        sub_img = frame[y : y + hud_h, x : x + hud_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.35, black_rect, 0.65, 0, sub_img)

        # Card border
        cv2.rectangle(frame, (x, y), (x + hud_w, y + hud_h), (80, 80, 80), 1, cv2.LINE_AA)

        # Text entries
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, model_name, (x + 10, y + 20), font, 0.50, (0, 220, 255), 1, cv2.LINE_AA)

        # FPS indicator (green if >= 30, yellow if 15-30, red if < 15)
        fps_color = (0, 255, 0) if fps >= 30 else ((0, 220, 255) if fps >= 15 else (0, 0, 255))
        cv2.putText(frame, f"FPS: {fps:4.1f}", (x + 10, y + 42), font, 0.48, fps_color, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"Latency: {inference_ms:4.1f} ms",
            (x + 10, y + 62),
            font,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Objects: {detection_count:<2d} | 'Q' to quit",
            (x + 10, y + 82),
            font,
            0.42,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        return frame
