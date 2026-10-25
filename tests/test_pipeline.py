"""
Unit tests for the integrated PPE Compliance Pipeline (detection/pipeline.py).

Tests:
  1. SimpleIoUTracker (lifecycle, matching, pruning, reset)
  2. TrackedPersonRecord schema & serialization (strict compliance with required keys)
  3. PPECompliancePipeline with modular mock components:
     - Full compliance (Hardhat + Vest)
     - Violations (Missing Hardhat, Missing Vest, Missing Both)
     - Unknown identity retention (never forcing identity)
     - Verified face identity assignment
     - Temporal anti-flicker smoothing across frames
     - Frame overlays & telemetry
     - Image & video processing paths
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from compliance.rules import ComplianceEngine, ComplianceStatus, DefaultPPE
from compliance.temporal_filter import PPETemporalFilter, TemporalFilterConfig
from detection.detector import Detection, DetectionResult, PPEDetector
from detection.pipeline import (
    ActiveTrack,
    PipelineResult,
    PipelineTelemetry,
    PPECompliancePipeline,
    SimpleIoUTracker,
    TrackedPersonRecord,
)
from detection.ppe_association import PPEAssociator
from recognition.registration import EmployeeRegistrationService


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def make_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a blank synthetic BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_person_det(
    x1: int = 100, y1: int = 50, x2: int = 300, y2: int = 450, conf: float = 0.90
) -> Detection:
    """Create a Person detection."""
    return Detection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        class_id=0,
        class_name="Person",
        confidence=conf,
    )


def make_ppe_det(
    class_name: str, x1: int, y1: int, x2: int, y2: int, conf: float = 0.85
) -> Detection:
    """Create a PPE detection."""
    cid_map = {"Hardhat": 1, "NO-Hardhat": 2, "Safety Vest": 3, "NO-Safety Vest": 4}
    return Detection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        class_id=cid_map.get(class_name, 99),
        class_name=class_name,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# Tracker Unit Tests
# ---------------------------------------------------------------------------


class TestSimpleIoUTracker:
    """Tests for SimpleIoUTracker spatial tracking."""

    def test_empty_detections(self):
        tracker = SimpleIoUTracker()
        matches = tracker.update([])
        assert matches == []

    def test_single_person_first_frame(self):
        tracker = SimpleIoUTracker()
        det = make_person_det(100, 100, 200, 300, 0.9)
        matches = tracker.update([det])

        assert len(matches) == 1
        track, matched_det = matches[0]
        assert track.track_id == "track_1"
        assert track.employee_id == "UNKNOWN"
        assert track.employee_name == "UNKNOWN"
        assert track.bbox == (100, 100, 200, 300)
        assert matched_det is det

    def test_tracking_across_frames_with_overlap(self):
        tracker = SimpleIoUTracker(iou_threshold=0.3)
        det_f1 = make_person_det(100, 100, 200, 300)
        matches_f1 = tracker.update([det_f1])
        assert matches_f1[0][0].track_id == "track_1"

        # Slight shift in frame 2
        det_f2 = make_person_det(105, 102, 205, 302)
        matches_f2 = tracker.update([det_f2])
        assert len(matches_f2) == 1
        assert matches_f2[0][0].track_id == "track_1"
        assert matches_f2[0][0].bbox == (105, 102, 205, 302)

    def test_multiple_people_tracking(self):
        tracker = SimpleIoUTracker()
        person_a = make_person_det(50, 50, 150, 250)
        person_b = make_person_det(300, 50, 400, 250)

        matches_f1 = tracker.update([person_a, person_b])
        assert len(matches_f1) == 2
        track_ids = {m[0].track_id for m in matches_f1}
        assert track_ids == {"track_1", "track_2"}

        # Next frame: shifted slightly
        person_a2 = make_person_det(55, 52, 155, 252)
        person_b2 = make_person_det(305, 52, 405, 252)
        matches_f2 = tracker.update([person_a2, person_b2])
        assert len(matches_f2) == 2
        track_ids_f2 = {m[0].track_id for m in matches_f2}
        assert track_ids_f2 == {"track_1", "track_2"}

    def test_pruning_dead_tracks(self):
        tracker = SimpleIoUTracker(max_missed=2)
        det = make_person_det(100, 100, 200, 300)
        tracker.update([det])
        assert "track_1" in tracker._tracks

        # Frame 2: person disappears
        tracker.update([])
        assert "track_1" in tracker._tracks  # Missed = 1 <= 2

        # Frame 3: person still missing
        tracker.update([])
        assert "track_1" in tracker._tracks  # Missed = 2 <= 2

        # Frame 4: missed = 3 > 2 -> pruned
        tracker.update([])
        assert "track_1" not in tracker._tracks

    def test_reset(self):
        tracker = SimpleIoUTracker()
        tracker.update([make_person_det(10, 10, 50, 100)])
        assert len(tracker._tracks) == 1
        tracker.reset()
        assert len(tracker._tracks) == 0
        assert tracker._next_id == 1


# ---------------------------------------------------------------------------
# TrackedPersonRecord Schema Tests
# ---------------------------------------------------------------------------


class TestTrackedPersonRecordSchema:
    """Verifies that TrackedPersonRecord adheres strictly to the required specification."""

    def test_required_keys_in_to_dict(self):
        rec = TrackedPersonRecord(
            track_id="track_1",
            employee_id="UNKNOWN",
            employee_name="UNKNOWN",
            hardhat=True,
            safety_vest=False,
            compliance_status="NON_COMPLIANT",
            missing_items=["Safety Vest"],
            detection_confidence=0.88765,
            bbox=(100, 50, 300, 450),
            face_similarity=None,
            is_compliant=False,
        )

        d = rec.to_dict()
        required_keys = {
            "track_id",
            "employee_id",
            "employee_name",
            "hardhat",
            "safety_vest",
            "compliance_status",
            "missing_items",
            "detection_confidence",
        }

        assert required_keys.issubset(d.keys())
        assert d["track_id"] == "track_1"
        assert d["employee_id"] == "UNKNOWN"
        assert d["employee_name"] == "UNKNOWN"
        assert d["hardhat"] is True
        assert d["safety_vest"] is False
        assert d["compliance_status"] == "NON_COMPLIANT"
        assert d["missing_items"] == ["Safety Vest"]
        assert d["detection_confidence"] == 0.8877

    def test_unknown_remains_strictly_unknown(self):
        rec = TrackedPersonRecord(
            track_id="track_42",
            employee_id="UNKNOWN",
            employee_name="UNKNOWN",
            hardhat=True,
            safety_vest=True,
            compliance_status="COMPLIANT",
            missing_items=[],
            detection_confidence=0.92,
        )
        d = rec.to_dict()
        assert d["employee_id"] == "UNKNOWN"
        assert d["employee_name"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Integrated Pipeline Tests
# ---------------------------------------------------------------------------


class TestPPECompliancePipeline:
    """Tests for the integrated PPE compliance pipeline."""

    def _build_mock_pipeline(
        self,
        mock_detections: Optional[List[Detection]] = None,
        mock_face_result: Optional[dict] = None,
    ) -> PPECompliancePipeline:
        """Helper to build a pipeline with mock detector and mock face service."""
        mock_detector = MagicMock(spec=PPEDetector)
        det_result = DetectionResult(
            detections=mock_detections or [],
            preprocess_ms=1.0,
            inference_ms=5.0,
            postprocess_ms=1.0,
            total_latency_ms=7.0,
            fps=140.0,
            orig_shape=(480, 640),
        )
        mock_detector.detect.return_value = det_result

        mock_face_service = MagicMock(spec=EmployeeRegistrationService)
        mock_face_service.identify.return_value = mock_face_result

        associator = PPEAssociator()
        temporal_filter = PPETemporalFilter(
            config=TemporalFilterConfig(window_size=3, min_observations=1)
        )
        compliance_engine = ComplianceEngine()
        tracker = SimpleIoUTracker()

        return PPECompliancePipeline(
            detector=mock_detector,
            associator=associator,
            face_service=mock_face_service,
            temporal_filter=temporal_filter,
            compliance_engine=compliance_engine,
            tracker=tracker,
        )

    def test_empty_frame_handling(self):
        pipeline = self._build_mock_pipeline()
        res = pipeline.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        assert res.total_tracked == 0
        assert res.records == []

    def test_full_compliance_single_person(self):
        """Person wearing both Hardhat and Safety Vest -> COMPLIANT."""
        detections = [
            make_person_det(100, 100, 300, 500, conf=0.95),
            make_ppe_det("Hardhat", 150, 105, 250, 160, conf=0.90),
            make_ppe_det("Safety Vest", 130, 200, 270, 380, conf=0.92),
        ]
        pipeline = self._build_mock_pipeline(mock_detections=detections)
        frame = make_dummy_frame()

        result = pipeline.process_frame(frame, draw_overlay=True)

        assert result.total_tracked == 1
        assert result.compliant_count == 1
        assert result.non_compliant_count == 0

        rec = result.records[0]
        assert rec.track_id == "track_1"
        assert rec.employee_id == "UNKNOWN"  # No face registered
        assert rec.employee_name == "UNKNOWN"
        assert rec.hardhat is True
        assert rec.safety_vest is True
        assert rec.compliance_status == "COMPLIANT"
        assert rec.missing_items == []
        assert rec.is_compliant is True
        assert result.annotated_frame is not None

    def test_non_compliance_missing_hardhat(self):
        """Person with NO-Hardhat and Safety Vest -> NON_COMPLIANT (Missing Hardhat)."""
        detections = [
            make_person_det(100, 100, 300, 500, conf=0.90),
            make_ppe_det("NO-Hardhat", 150, 105, 250, 160, conf=0.88),
            make_ppe_det("Safety Vest", 130, 200, 270, 380, conf=0.91),
        ]
        pipeline = self._build_mock_pipeline(mock_detections=detections)
        frame = make_dummy_frame()

        result = pipeline.process_frame(frame)
        rec = result.records[0]

        assert rec.hardhat is False
        assert rec.safety_vest is True
        assert rec.compliance_status == "NON_COMPLIANT"
        assert "Hardhat" in rec.missing_items
        assert "Safety Vest" not in rec.missing_items
        assert rec.is_compliant is False

    def test_non_compliance_missing_safety_vest(self):
        """Person with Hardhat and NO-Safety Vest -> NON_COMPLIANT (Missing Safety Vest)."""
        detections = [
            make_person_det(100, 100, 300, 500, conf=0.90),
            make_ppe_det("Hardhat", 150, 105, 250, 160, conf=0.88),
            make_ppe_det("NO-Safety Vest", 130, 200, 270, 380, conf=0.85),
        ]
        pipeline = self._build_mock_pipeline(mock_detections=detections)
        frame = make_dummy_frame()

        result = pipeline.process_frame(frame)
        rec = result.records[0]

        assert rec.hardhat is True
        assert rec.safety_vest is False
        assert rec.compliance_status == "NON_COMPLIANT"
        assert "Safety Vest" in rec.missing_items
        assert "Hardhat" not in rec.missing_items
        assert rec.is_compliant is False

    def test_non_compliance_missing_both(self):
        """Person with neither Hardhat nor Safety Vest -> NON_COMPLIANT (Both missing)."""
        detections = [
            make_person_det(100, 100, 300, 500, conf=0.90),
            make_ppe_det("NO-Hardhat", 150, 105, 250, 160, conf=0.88),
            make_ppe_det("NO-Safety Vest", 130, 200, 270, 380, conf=0.85),
        ]
        pipeline = self._build_mock_pipeline(mock_detections=detections)
        frame = make_dummy_frame()

        result = pipeline.process_frame(frame)
        rec = result.records[0]

        assert rec.hardhat is False
        assert rec.safety_vest is False
        assert rec.compliance_status == "NON_COMPLIANT"
        assert set(rec.missing_items) == {"Hardhat", "Safety Vest"}
        assert rec.is_compliant is False

    def test_face_recognition_verified_employee(self):
        """Verified face match assigns identity to track."""
        detections = [
            make_person_det(100, 100, 300, 500, conf=0.92),
            make_ppe_det("Hardhat", 150, 105, 250, 160, conf=0.88),
            make_ppe_det("Safety Vest", 130, 200, 270, 380, conf=0.91),
        ]
        mock_face_match = {
            "employee_id": "EMP101",
            "employee_name": "Alice Smith",
            "similarity": 0.72,
        }
        pipeline = self._build_mock_pipeline(
            mock_detections=detections,
            mock_face_result=mock_face_match,
        )
        frame = make_dummy_frame()

        result = pipeline.process_frame(frame)
        rec = result.records[0]

        assert rec.employee_id == "EMP101"
        assert rec.employee_name == "Alice Smith"
        assert rec.face_similarity == 0.72
        assert rec.is_compliant is True

    def test_unknown_people_strictly_remain_unknown(self):
        """When face recognition returns None (no match / low confidence), person remains UNKNOWN."""
        detections = [
            make_person_det(100, 100, 300, 500, conf=0.88),
        ]
        pipeline = self._build_mock_pipeline(
            mock_detections=detections,
            mock_face_result=None,  # Face rejected or unrecognized
        )
        frame = make_dummy_frame()

        result = pipeline.process_frame(frame)
        rec = result.records[0]

        assert rec.employee_id == "UNKNOWN"
        assert rec.employee_name == "UNKNOWN"
        assert rec.face_similarity is None

    def test_temporal_smoothing_suppresses_single_bad_frame(self):
        """Single missing-frame fluctuation does not immediately change compliant status."""
        frame = make_dummy_frame()

        # Window size 5, presence_threshold 0.60
        comp_engine = ComplianceEngine()
        filter_5 = PPETemporalFilter(
            config=TemporalFilterConfig(
                window_size=5,
                presence_threshold=0.60,
                min_observations=1,
            ),
            compliance_engine=comp_engine,
        )

        mock_det = MagicMock(spec=PPEDetector)
        pipeline = PPECompliancePipeline(
            detector=mock_det,
            temporal_filter=filter_5,
            face_service=None,  # skip face
        )

        # Frames 1-3: Fully compliant
        compliant_dets = [
            make_person_det(100, 100, 300, 500),
            make_ppe_det("Hardhat", 150, 105, 250, 160),
            make_ppe_det("Safety Vest", 130, 200, 270, 380),
        ]
        mock_det.detect.return_value = DetectionResult(detections=compliant_dets)

        for _ in range(3):
            res = pipeline.process_frame(frame)
            assert res.records[0].compliance_status == "COMPLIANT"

        # Frame 4: Bad frame (Hardhat detection temporarily dropped/flickered)
        flicker_dets = [
            make_person_det(100, 100, 300, 500),
            make_ppe_det("Safety Vest", 130, 200, 270, 380),
        ]
        mock_det.detect.return_value = DetectionResult(detections=flicker_dets)
        res_flicker = pipeline.process_frame(frame)

        # 3 out of 4 frames had hardhat (75% >= 60%) -> stable state remains COMPLIANT
        assert res_flicker.records[0].compliance_status == "COMPLIANT"
        assert res_flicker.records[0].hardhat is True

    def test_process_image_convenience(self):
        """Tests pipeline.process_image with a numpy array input."""
        detections = [
            make_person_det(50, 50, 200, 400),
            make_ppe_det("Hardhat", 80, 55, 170, 110),
            make_ppe_det("Safety Vest", 70, 120, 180, 260),
        ]
        pipeline = self._build_mock_pipeline(mock_detections=detections)
        frame = make_dummy_frame()

        res = pipeline.process_image(frame, draw_overlay=True)
        assert res.total_tracked == 1
        assert res.annotated_frame is not None
        assert res.records[0].compliance_status == "COMPLIANT"

    def test_pipeline_result_to_dict(self):
        """Ensure PipelineResult.to_dict formats properly."""
        detections = [
            make_person_det(50, 50, 200, 400),
            make_ppe_det("Hardhat", 80, 55, 170, 110),
            make_ppe_det("Safety Vest", 70, 120, 180, 260),
        ]
        pipeline = self._build_mock_pipeline(mock_detections=detections)
        frame = make_dummy_frame()

        res = pipeline.process_frame(frame)
        d = res.to_dict()

        assert d["total_tracked"] == 1
        assert d["compliant_count"] == 1
        assert d["non_compliant_count"] == 0
        assert len(d["persons"]) == 1
        assert d["persons"][0]["track_id"] == "track_1"
        assert "telemetry" in d
        assert "fps" in d["telemetry"]

    def test_multiple_persons_never_assigned_duplicate_identity(self):
        """When two persons both match the same employee, only the best match retains it."""
        detections = [
            make_person_det(50, 50, 200, 400, conf=0.95),
            make_person_det(250, 50, 400, 400, conf=0.90),
        ]
        mock_face_service = MagicMock(spec=EmployeeRegistrationService)
        # First call has higher similarity (0.85), second call has lower similarity (0.60), both claim EMP101
        mock_face_service.identify.side_effect = [
            {"employee_id": "EMP101", "employee_name": "Alice Smith", "similarity": 0.85},
            {"employee_id": "EMP101", "employee_name": "Alice Smith", "similarity": 0.60},
        ]
        mock_detector = MagicMock(spec=PPEDetector)
        mock_detector.detect.return_value = DetectionResult(detections=detections)

        pipeline = PPECompliancePipeline(
            detector=mock_detector,
            face_service=mock_face_service,
        )
        frame = make_dummy_frame()
        res = pipeline.process_frame(frame)

        assert res.total_tracked == 2
        assigned_ids = [r.employee_id for r in res.records]
        assert assigned_ids.count("EMP101") == 1
        assert assigned_ids.count("UNKNOWN") == 1

    def test_reverification_grace_period_and_revocation(self):
        """Transient missed faces preserve identity; 3 consecutive failures revoke to UNKNOWN."""
        det = make_person_det(50, 50, 200, 400, conf=0.95)
        mock_detector = MagicMock(spec=PPEDetector)
        mock_detector.detect.return_value = DetectionResult(detections=[det])

        mock_face_service = MagicMock(spec=EmployeeRegistrationService)
        call_count = 0

        def mock_identify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call succeeds as Alice
            if call_count == 1:
                return {"employee_id": "EMP101", "employee_name": "Alice Smith", "similarity": 0.85}
            # All subsequent re-checks return None (missed face)
            return None

        mock_face_service.identify.side_effect = mock_identify

        # Use face_check_interval=1 so verified recheck_interval = 6 frames
        pipeline = PPECompliancePipeline(
            detector=mock_detector,
            face_service=mock_face_service,
            face_check_interval=1,
        )
        frame = make_dummy_frame()

        # Frame 0: Identified as Alice
        res0 = pipeline.process_frame(frame)
        assert res0.records[0].employee_id == "EMP101"

        # Advance 6 frames to trigger 1st re-check (returns None)
        for _ in range(6):
            res = pipeline.process_frame(frame)
        # 1st failed recheck: Grace period keeps Alice
        assert res.records[0].employee_id == "EMP101"

        # Advance another 6 frames to trigger 2nd re-check (returns None)
        for _ in range(6):
            res = pipeline.process_frame(frame)
        # 2nd failed recheck: Grace period still keeps Alice
        assert res.records[0].employee_id == "EMP101"

        # Advance another 6 frames to trigger 3rd re-check (returns None)
        for _ in range(6):
            res = pipeline.process_frame(frame)
        # 3rd failed recheck: Revoked back to UNKNOWN
        assert res.records[0].employee_id == "UNKNOWN"

