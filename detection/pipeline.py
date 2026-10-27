"""
PPE Compliance & Real-Time Monitoring Integration Pipeline.

Unifies all modular system components:
  1. Person & PPE Object Detection (YOLOv8)
  2. Multi-Object Person Tracking (IoU-based spatial tracking)
  3. Person-to-PPE Spatial Association (Anatomical priors)
  4. Face Recognition & Identity Verification (YuNet + SFace)
  5. Temporal Smoothing & Anti-Flicker Filtering (Sliding-window voting)
  6. Safety Compliance Evaluation (Rule engine)

For every tracked person in a frame, outputs a structured record:
{
    "track_id": "track_1",
    "employee_id": "EMP001",      # Or "UNKNOWN"
    "employee_name": "Jane Doe",  # Or "UNKNOWN"
    "hardhat": True,
    "safety_vest": True,
    "compliance_status": "COMPLIANT",
    "missing_items": [],
    "detection_confidence": 0.95
}
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

# Ensure project root is importable
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compliance.rules import ComplianceEngine, ComplianceStatus, PersonPPEInfo
from compliance.temporal_filter import PPETemporalFilter, SmoothedPersonState
from detection.detector import Detection, PPEDetector
from detection.ppe_association import AssociatedPerson, AssociationResult, PPEAssociator
from recognition.registration import EmployeeRegistrationService, EmployeeStorage

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Structured Pipeline Output
# ---------------------------------------------------------------------------


@dataclass
class TrackedPersonRecord:
    """Consolidated record for a tracked person combining identity and PPE compliance."""

    track_id: str
    employee_id: str  # "EMP001" or "UNKNOWN"
    employee_name: str  # "Jane Doe" or "UNKNOWN"
    hardhat: bool
    safety_vest: bool
    compliance_status: str  # "COMPLIANT" or "NON_COMPLIANT"
    missing_items: List[str]
    detection_confidence: float
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    face_similarity: Optional[float] = None
    is_compliant: bool = False
    raw_ppe: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to the required structured dictionary."""
        return {
            "track_id": self.track_id,
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "hardhat": self.hardhat,
            "safety_vest": self.safety_vest,
            "compliance_status": self.compliance_status,
            "missing_items": list(self.missing_items),
            "detection_confidence": round(float(self.detection_confidence), 4),
            "bbox": list(self.bbox),
            "is_compliant": self.is_compliant,
            "face_similarity": (
                round(float(self.face_similarity), 4)
                if self.face_similarity is not None
                else None
            ),
        }


@dataclass
class PipelineTelemetry:
    """Latency and performance metrics for a processed frame."""

    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    association_ms: float = 0.0
    recognition_ms: float = 0.0
    smoothing_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0


@dataclass
class PipelineResult:
    """Result of processing a single frame or test image."""

    records: List[TrackedPersonRecord] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    telemetry: PipelineTelemetry = field(default_factory=PipelineTelemetry)
    unassigned_ppe: List[Detection] = field(default_factory=list)

    @property
    def total_tracked(self) -> int:
        return len(self.records)

    @property
    def compliant_count(self) -> int:
        return sum(1 for r in self.records if r.is_compliant)

    @property
    def non_compliant_count(self) -> int:
        return self.total_tracked - self.compliant_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete frame results to dictionary."""
        return {
            "total_tracked": self.total_tracked,
            "compliant_count": self.compliant_count,
            "non_compliant_count": self.non_compliant_count,
            "persons": [r.to_dict() for r in self.records],
            "telemetry": {
                "total_ms": self.telemetry.total_ms,
                "fps": self.telemetry.fps,
                "detection_ms": self.telemetry.detection_ms,
                "association_ms": self.telemetry.association_ms,
                "recognition_ms": self.telemetry.recognition_ms,
                "smoothing_ms": self.telemetry.smoothing_ms,
            },
        }


# ---------------------------------------------------------------------------
# Simple Multi-Object Spatial Tracker (IoU Tracker)
# ---------------------------------------------------------------------------


@dataclass
class ActiveTrack:
    """Represents a continuous track of a person across frames."""

    track_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    last_seen_frame: int
    employee_id: str = "UNKNOWN"
    employee_name: str = "UNKNOWN"
    face_similarity: Optional[float] = None
    face_attempts: int = 0
    failed_rechecks: int = 0


class SimpleIoUTracker:
    """Lightweight, deterministic IoU spatial tracker for person bounding boxes."""

    def __init__(self, iou_threshold: float = 0.30, max_missed: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: Dict[str, ActiveTrack] = {}
        self._frame_count = 0

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, person_detections: List[Detection]) -> List[Tuple[ActiveTrack, Detection]]:
        """Match incoming detections to existing tracks and spawn new tracks.

        Returns:
            List of matched (ActiveTrack, Detection) pairs for current frame.
        """
        self._frame_count += 1
        matched_pairs: List[Tuple[ActiveTrack, Detection]] = []

        active_track_ids = list(self._tracks.keys())
        assigned_det_indices: Set[int] = set()
        assigned_track_ids: Set[str] = set()

        if active_track_ids and person_detections:
            # Pairwise IoU
            iou_matrix = np.zeros((len(active_track_ids), len(person_detections)), dtype=np.float32)
            for t_idx, tid in enumerate(active_track_ids):
                for d_idx, det in enumerate(person_detections):
                    iou_matrix[t_idx, d_idx] = self._compute_iou(self._tracks[tid].bbox, det.box)

            # Greedy matching by maximum IoU
            while True:
                max_iou = float(np.max(iou_matrix)) if iou_matrix.size > 0 else 0.0
                if max_iou < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = active_track_ids[t_idx]

                track = self._tracks[tid]
                det = person_detections[d_idx]

                track.bbox = det.box
                track.confidence = det.confidence
                track.last_seen_frame = self._frame_count

                matched_pairs.append((track, det))
                assigned_det_indices.add(d_idx)
                assigned_track_ids.add(tid)

                # Nullify row and column
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(person_detections):
            if d_idx not in assigned_det_indices:
                tid = f"track_{self._next_id}"
                self._next_id += 1
                new_track = ActiveTrack(
                    track_id=tid,
                    bbox=det.box,
                    confidence=det.confidence,
                    last_seen_frame=self._frame_count,
                )
                self._tracks[tid] = new_track
                matched_pairs.append((new_track, det))

        # Prune dead tracks
        dead_ids = [
            tid
            for tid, t in self._tracks.items()
            if (self._frame_count - t.last_seen_frame) > self.max_missed
        ]
        for tid in dead_ids:
            del self._tracks[tid]

        return matched_pairs

    def reset(self) -> None:
        """Reset all active tracks and counter."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0


# ---------------------------------------------------------------------------
# Integrated Compliance Pipeline
# ---------------------------------------------------------------------------


class PPECompliancePipeline:
    """Full-stack integration pipeline linking detection, tracking, association,
    face recognition, temporal smoothing, and compliance rules.
    """

    def __init__(
        self,
        detector: Optional[PPEDetector] = None,
        associator: Optional[PPEAssociator] = None,
        face_service: Optional[EmployeeRegistrationService] = None,
        temporal_filter: Optional[PPETemporalFilter] = None,
        compliance_engine: Optional[ComplianceEngine] = None,
        tracker: Optional[SimpleIoUTracker] = None,
        face_confidence_threshold: float = 0.50,
        face_check_interval: int = 10,
    ) -> None:
        """Initialize the integrated pipeline.

        Args:
            detector: Object detection engine (YOLOv8).
            associator: Person-to-PPE geometric associator.
            face_service: Employee face registration and identification service.
            temporal_filter: Multi-frame temporal smoothing filter.
            compliance_engine: Safety equipment rule engine.
            tracker: Multi-object person spatial tracker.
            face_confidence_threshold: Minimum similarity to confirm identity.
            face_check_interval: How often (in frames) to re-evaluate face identity per track.
        """
        # Lazy/default component instantiation
        if detector is None:
            from detection.detector import DEFAULT_ONNX_MODEL, resolve_detector_backend

            if resolve_detector_backend() == "onnx":
                detector = PPEDetector(model_path=DEFAULT_ONNX_MODEL, device="cpu")
            else:
                detector = PPEDetector(device="auto")
        self.detector = detector
        self.associator = associator or PPEAssociator()
        self.face_service = face_service or EmployeeRegistrationService()
        self.compliance_engine = compliance_engine or ComplianceEngine()
        self.temporal_filter = temporal_filter or PPETemporalFilter(
            compliance_engine=self.compliance_engine
        )
        self.tracker = tracker or SimpleIoUTracker()

        self.face_confidence_threshold = float(face_confidence_threshold)
        self.face_check_interval = int(face_check_interval)

    # -----------------------------------------------------------------------
    # Core Processing
    # -----------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        draw_overlay: bool = True,
        draw_hud: bool = False,
    ) -> PipelineResult:
        """Process a single frame through the full integration pipeline.

        Args:
            frame: Input image in BGR format.
            frame_id: Optional sequential frame index.
            draw_overlay: If True, draws visual bounding boxes and telemetry.
            draw_hud: If True, draws telemetry HUD box in top-left (default False).

        Returns:
            PipelineResult containing list of TrackedPersonRecord and annotated frame.
        """
        t_start = time.perf_counter()
        if frame is None or frame.size == 0:
            return PipelineResult()

        h, w = frame.shape[:2]

        # -------------------------------------------------------------------
        # 1. Detection (Person + PPE)
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        det_result = self.detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000.0

        # Separate Person vs PPE detections
        person_detections: List[Detection] = []
        ppe_detections: List[Detection] = []

        for d in det_result.detections:
            if d.class_name == "Person":
                person_detections.append(d)
            else:
                ppe_detections.append(d)

        # -------------------------------------------------------------------
        # 2. Tracking (Multi-object spatial tracking)
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        matched_tracks = self.tracker.update(person_detections)
        t_track = (time.perf_counter() - t0) * 1000.0

        # Keep active tracker IDs in temporal filter to prune departed tracks
        active_ids = {track.track_id for track, _ in matched_tracks}
        self.temporal_filter.remove_inactive(active_ids)

        # -------------------------------------------------------------------
        # 3. Spatial Person-to-PPE Association
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        tracked_person_detections = [det for _, det in matched_tracks]
        assoc_result: AssociationResult = self.associator.associate(
            tracked_person_detections,
            ppe_detections,
        )
        t_assoc = (time.perf_counter() - t0) * 1000.0

        # ---------------------------------------------------------------
        # 4. Face Recognition & Identity Verification
        # ---------------------------------------------------------------
        t_rec_accum = 0.0
        for track, _ in matched_tracks:
            t_rec_start = time.perf_counter()
            # Adaptive face verification:
            # Check immediately on frame 0, and every 2 frames while UNKNOWN so identity is discovered promptly
            # If verified: re-verify periodically (every 60 frames ~ 2s) to maintain continuous tracking
            recheck_interval = self.face_check_interval * 6 if track.employee_id != "UNKNOWN" else 2
            should_check_face = (track.face_attempts == 0) or (track.face_attempts % recheck_interval == 0)
            track.face_attempts += 1

            if should_check_face and self.face_service:
                px1, py1, px2, py2 = track.bbox
                c_px1 = max(0, min(w - 1, px1))
                c_py1 = max(0, min(h - 1, py1))
                c_px2 = max(c_px1 + 1, min(w, px2))
                c_py2 = max(c_py1 + 1, min(h, py2))

                # Upper-body / head crop (upper 65% covers head, face, and chin for seated & standing postures)
                head_h = int(max(40, (py2 - py1) * 0.65))
                c_head_y2 = max(c_py1 + 1, min(h, c_py1 + head_h))
                head_crop = frame[c_py1:c_head_y2, c_px1:c_px2]

                match = None
                if head_crop.size > 0:
                    match = self.face_service.identify(
                        head_crop,
                        threshold=self.face_confidence_threshold,
                        prefer_center=True,
                    )

                # Fallback to full person crop if upper crop didn't find face (e.g. close-up webcam)
                if match is None:
                    person_crop = frame[c_py1:c_py2, c_px1:c_px2]
                    if person_crop.size > 0:
                        match = self.face_service.identify(
                            person_crop,
                            threshold=self.face_confidence_threshold,
                            prefer_center=True,
                        )

                if match is not None:
                    # Verified identity
                    track.employee_id = match["employee_id"]
                    track.employee_name = match["employee_name"]
                    track.face_similarity = match["similarity"]
                    track.failed_rechecks = 0
                else:
                    # Revoke stale identity on multiple consecutive re-verification failures.
                    # Prevents transient occlusions or head turns from flickering identity to UNKNOWN,
                    # while ensuring track identity is revoked if the person actually leaves.
                    if track.employee_id != "UNKNOWN":
                        track.failed_rechecks += 1
                        if track.failed_rechecks >= 3:
                            track.employee_id = "UNKNOWN"
                            track.employee_name = "UNKNOWN"
                            track.face_similarity = None
                            track.failed_rechecks = 0

            t_rec_accum += (time.perf_counter() - t_rec_start) * 1000.0

        # Enforce 1-to-1 identity uniqueness across all tracks in the current frame:
        # A single human employee cannot be in multiple places at the same time.
        assigned_employees: Dict[str, Tuple[ActiveTrack, float]] = {}
        for track, _ in matched_tracks:
            if track.employee_id != "UNKNOWN":
                sim = track.face_similarity if track.face_similarity is not None else 0.0
                if track.employee_id in assigned_employees:
                    prev_track, prev_sim = assigned_employees[track.employee_id]
                    # Retain identity for the track with higher similarity score; reset duplicate to UNKNOWN
                    if sim > prev_sim:
                        prev_track.employee_id = "UNKNOWN"
                        prev_track.employee_name = "UNKNOWN"
                        prev_track.face_similarity = None
                        assigned_employees[track.employee_id] = (track, sim)
                    else:
                        track.employee_id = "UNKNOWN"
                        track.employee_name = "UNKNOWN"
                        track.face_similarity = None
                else:
                    assigned_employees[track.employee_id] = (track, sim)

        # ---------------------------------------------------------------
        # 5. Temporal Smoothing & Anti-Flicker Filtering
        # ---------------------------------------------------------------
        person_records: List[TrackedPersonRecord] = []
        t_smooth_accum = 0.0

        for (track, _), assoc_person in zip(matched_tracks, assoc_result.persons):
            t_sm_start = time.perf_counter()

            # Build single-frame PPE info from spatial association
            raw_ppe_info = assoc_person.to_person_ppe_info()
            raw_ppe_info.person_id = track.track_id

            # Apply sliding-window temporal smoothing
            smoothed_state: SmoothedPersonState = self.temporal_filter.update(raw_ppe_info)

            t_smooth_accum += (time.perf_counter() - t_sm_start) * 1000.0

            # ---------------------------------------------------------------
            # 6. Structured Record Creation
            # ---------------------------------------------------------------
            rec = TrackedPersonRecord(
                track_id=track.track_id,
                employee_id=track.employee_id,
                employee_name=track.employee_name,
                hardhat=smoothed_state.has_hardhat,
                safety_vest=smoothed_state.has_safety_vest,
                compliance_status=smoothed_state.status.value,
                missing_items=list(smoothed_state.missing_equipment),
                detection_confidence=track.confidence,
                bbox=track.bbox,
                face_similarity=track.face_similarity,
                is_compliant=smoothed_state.is_compliant,
                raw_ppe={
                    "hardhat_count": len(assoc_person.hardhats),
                    "no_hardhat_count": len(assoc_person.no_hardhats),
                    "vest_count": len(assoc_person.safety_vests),
                    "no_vest_count": len(assoc_person.no_safety_vests),
                },
            )
            person_records.append(rec)

        t_total = (time.perf_counter() - t_start) * 1000.0
        fps = round(1000.0 / t_total, 1) if t_total > 0 else 0.0

        telemetry = PipelineTelemetry(
            detection_ms=round(t_det, 2),
            tracking_ms=round(t_track, 2),
            association_ms=round(t_assoc, 2),
            recognition_ms=round(t_rec_accum, 2),
            smoothing_ms=round(t_smooth_accum, 2),
            total_ms=round(t_total, 2),
            fps=fps,
        )

        annotated = None
        if draw_overlay:
            annotated = self.render_overlay(frame.copy(), person_records, telemetry, draw_hud=draw_hud)

        return PipelineResult(
            records=person_records,
            annotated_frame=annotated,
            telemetry=telemetry,
            unassigned_ppe=assoc_result.unassigned_ppe,
        )

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        draw_overlay: bool = True,
    ) -> PipelineResult:
        """Process a static image file or numpy array through the pipeline.

        Args:
            image_input: File path or BGR image array.
            draw_overlay: Whether to generate an annotated frame.

        Returns:
            PipelineResult containing records and annotated image.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                logger.error("Failed to load image: %s", image_input)
                return PipelineResult()
        else:
            img = image_input

        # For static image evaluation, reset tracker to start fresh
        self.tracker.reset()
        return self.process_frame(img, draw_overlay=draw_overlay)

    # -----------------------------------------------------------------------
    # Visual Overlay & HUD
    # -----------------------------------------------------------------------

    def render_overlay(
        self,
        frame: np.ndarray,
        records: List[TrackedPersonRecord],
        telemetry: PipelineTelemetry,
        draw_hud: bool = False,
    ) -> np.ndarray:
        """Render person compliance bounding boxes, identity tags, and optional telemetry HUD."""
        out = frame

        for rec in records:
            x1, y1, x2, y2 = rec.bbox

            # Border color: Green if Compliant, Red if Non-Compliant
            box_color = (0, 200, 0) if rec.is_compliant else (0, 0, 230)

            # Draw person bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Identity & compliance label text
            # e.g.: "[EMP001: Jane Doe] COMPLIANT"
            #   or: "[UNKNOWN] NON-COMPLIANT (No: Hardhat)"
            id_str = (
                f"{rec.employee_id}: {rec.employee_name}"
                if rec.employee_id != "UNKNOWN"
                else "UNKNOWN"
            )

            status_str = "COMPLIANT" if rec.is_compliant else f"NON-COMPLIANT ({', '.join(rec.missing_items)})"
            label = f"[{id_str}] {status_str}"

            # Calculate text size for label pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.45, 1)

            bx1 = x1
            by1 = max(0, y1 - th - baseline - 4)
            bx2 = x1 + tw + 6
            by2 = y1

            if by1 < 0:
                by1 = y1
                by2 = y1 + th + baseline + 4

            # Draw label background pill
            cv2.rectangle(out, (bx1, by1), (bx2, by2), box_color, cv2.FILLED)
            cv2.putText(
                out,
                label,
                (bx1 + 3, by2 - baseline - 1),
                font,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if not draw_hud:
            return out

        # Draw HUD in top-left
        hud_w, hud_h = 280, 110
        x, y = 12, 12

        sub_img = out[y : y + hud_h, x : x + hud_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.35, black_rect, 0.65, 0, sub_img)
        cv2.rectangle(out, (x, y), (x + hud_w, y + hud_h), (80, 80, 80), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, "PPE COMPLIANCE PIPELINE", (x + 10, y + 20), font, 0.50, (0, 220, 255), 1, cv2.LINE_AA)

        fps_col = (0, 255, 0) if telemetry.fps >= 30 else (0, 220, 255)
        cv2.putText(out, f"FPS: {telemetry.fps:4.1f} ({telemetry.total_ms:4.1f} ms)", (x + 10, y + 42), font, 0.45, fps_col, 1, cv2.LINE_AA)

        compliant_n = sum(1 for r in records if r.is_compliant)
        non_comp_n = len(records) - compliant_n
        cv2.putText(
            out,
            f"Workers: {len(records)} | Compliant: {compliant_n} | Violations: {non_comp_n}",
            (x + 10, y + 64),
            font,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            f"Det: {telemetry.detection_ms:.1f}ms | Face: {telemetry.recognition_ms:.1f}ms",
            (x + 10, y + 84),
            font,
            0.40,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            "Press 'Q' to quit",
            (x + 10, y + 102),
            font,
            0.38,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the integration pipeline."""
    parser = argparse.ArgumentParser(
        description="Run integrated real-time PPE compliance pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Path to single test image.")
    parser.add_argument("--video", type=str, default=None, help="Path to test video file.")
    parser.add_argument("--camera", type=str, default=None, help="Webcam device index (e.g. 0).")
    parser.add_argument("--model", type=str, default="models/weights/best.pt", help="YOLO model path.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection threshold.")
    parser.add_argument("--face-thresh", type=float, default=0.50, help="Face recognition threshold.")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: 'auto', '0', 'cpu'.")
    parser.add_argument("--flip-h", action="store_true", default=True, help="Flip camera horizontally (mirror mode).")
    parser.add_argument("--no-flip-h", dest="flip_h", action="store_false", help="Disable horizontal flip.")
    parser.add_argument("--flip-v", action="store_true", default=False, help="Flip camera vertically.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display.")
    return parser


def main() -> None:
    """CLI runner for pipeline on webcam, video, or image."""
    parser = build_parser()
    args = parser.parse_args()
    detector = PPEDetector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    pipeline = PPECompliancePipeline(
        detector=detector,
        face_confidence_threshold=args.face_thresh,
    )

    # 1. Image Mode
    if args.image:
        print(f"\nProcessing test image: {args.image}")
        res = pipeline.process_image(args.image, draw_overlay=True)

        print("\n" + "=" * 60)
        print("  PPE COMPLIANCE RESULTS")
        print("=" * 60)
        for rec in res.records:
            print(f"  Track ID    : {rec.track_id}")
            print(f"  Employee    : {rec.employee_id} ({rec.employee_name})")
            print(f"  Hardhat     : {'YES' if rec.hardhat else 'NO'}")
            print(f"  Safety Vest : {'YES' if rec.safety_vest else 'NO'}")
            print(f"  Compliance  : {rec.compliance_status}")
            print(f"  Missing     : {rec.missing_items or 'None'}")
            print("-" * 60)

        print(f"Total Workers: {res.total_tracked} | Compliant: {res.compliant_count} | Violations: {res.non_compliant_count}")
        print(f"Total Latency: {res.telemetry.total_ms:.1f} ms\n")

        if args.save_output and res.annotated_frame is not None:
            cv2.imwrite(args.save_output, res.annotated_frame)
            print(f"Saved annotated image to: {args.save_output}")
        return

    # 2. Camera or Video Stream Mode
    camera_source: Union[int, str] = 0
    if args.camera is not None:
        camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    elif args.video is not None:
        camera_source = args.video

    print(f"\nOpening stream source: {camera_source}")
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{camera_source}'")
        sys.exit(1)

    window_name = "Real-Time PPE Compliance Pipeline — 'Q' to Exit"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if args.flip_h and args.flip_v:
                frame = cv2.flip(frame, -1)
            elif args.flip_h:
                frame = cv2.flip(frame, 1)
            elif args.flip_v:
                frame = cv2.flip(frame, 0)

            result = pipeline.process_frame(frame, draw_overlay=True)

            if not args.headless and result.annotated_frame is not None:
                cv2.imshow(window_name, result.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
