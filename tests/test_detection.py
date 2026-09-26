"""Unit tests for detection/detector.py and detection/webcam.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detection.detector import (
    CLASS_COLORS_BGR,
    Detection,
    DetectionResult,
    PPEDetector,
)
from detection.webcam import (
    build_parser,
    open_camera,
    parse_camera_source,
    run_webcam,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestDetectionContainers:
    """Test Detection and DetectionResult data structures."""

    def test_detection_properties(self):
        d = Detection(
            x1=10,
            y1=20,
            x2=110,
            y2=220,
            class_id=2,
            class_name="Hardhat",
            confidence=0.92,
        )
        assert d.box == (10, 20, 110, 220)
        assert d.width == 100
        assert d.height == 200
        assert d.area == 20000
        assert d.center == (60, 120)

    def test_detection_result_filter_by_class(self):
        d1 = Detection(0, 0, 10, 10, 0, "Person", 0.9)
        d2 = Detection(0, 0, 10, 10, 1, "Hardhat", 0.85)
        d3 = Detection(0, 0, 10, 10, 2, "NO-Hardhat", 0.75)
        res = DetectionResult(detections=[d1, d2, d3])

        assert res.count == 3
        hardhats = res.filter_by_class(["Hardhat"])
        assert len(hardhats) == 1
        assert hardhats[0].class_name == "Hardhat"

        persons = res.filter_by_class(["Person", "Hardhat"])
        assert len(persons) == 2


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


class TestPPEDetector:
    """Test PPEDetector initialization and inference handling."""

    def test_missing_model_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            PPEDetector(model_path="nonexistent_model_file.pt")
        assert "not found" in str(exc_info.value).lower()

    def test_class_colors_defined_for_ppe(self):
        for name in ["Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"]:
            color = PPEDetector.get_class_color(name)
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_draw_detections_renders_boxes(self):
        # Create a dummy image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(50, 50, 200, 200, 0, "Hardhat", 0.95),
            Detection(100, 100, 300, 400, 1, "Person", 0.88),
        ]
        detector = MagicMock(spec=PPEDetector)
        detector.get_class_color = PPEDetector.get_class_color

        annotated = PPEDetector.draw_detections(detector, frame, detections, copy=True)
        assert annotated.shape == (480, 640, 3)
        # Check that frame was drawn upon (contains non-zero pixels)
        assert np.any(annotated > 0)

    def test_draw_hud_renders_overlay(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = MagicMock(spec=PPEDetector)
        out = PPEDetector.draw_hud(detector, frame, fps=60.0, inference_ms=5.5, detection_count=3)
        assert out.shape == (480, 640, 3)
        assert np.any(out > 0)


# ---------------------------------------------------------------------------
# Webcam CLI & helper tests
# ---------------------------------------------------------------------------


class TestWebcamHelpers:
    """Test webcam argument parsing and helper functions."""

    def test_parse_camera_source_int(self):
        assert parse_camera_source("0") == 0
        assert parse_camera_source("2") == 2
        assert parse_camera_source(" 1 ") == 1

    def test_parse_camera_source_string(self):
        assert parse_camera_source("video.mp4") == "video.mp4"
        assert parse_camera_source("rtsp://192.168.1.1/live") == "rtsp://192.168.1.1/live"

    def test_open_camera_invalid_source_returns_none(self):
        cap = open_camera(-999)
        assert cap is None

    def test_webcam_cli_parser(self):
        parser = build_parser()
        args = parser.parse_args(["--camera", "1", "--confidence", "0.4", "--width", "1280", "--height", "720"])
        assert args.camera == "1"
        assert args.confidence == 0.4
        assert args.width == 1280
        assert args.height == 720
        assert args.model == "models/weights/best.pt"
        assert args.device == "auto"
        assert args.headless is False

    @patch("detection.webcam.open_camera")
    @patch("detection.webcam.PPEDetector")
    def test_run_webcam_processes_frames(self, mock_detector_cls, mock_open_camera):
        # Mock detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = DetectionResult(
            detections=[Detection(10, 10, 50, 50, 0, "Person", 0.9)],
            inference_ms=3.0,
        )
        mock_detector_cls.return_value = mock_detector

        # Mock capture with 3 frames
        mock_cap = MagicMock()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, dummy_frame),
            (True, dummy_frame),
            (True, dummy_frame),
            (False, None),
        ]
        mock_cap.isOpened.return_value = True
        mock_open_camera.return_value = mock_cap

        processed = run_webcam(
            camera_source=0,
            model_path="models/weights/best.pt",
            headless=True,
            max_frames=3,
        )

        assert processed == 3
        assert mock_detector.detect.call_count == 3
        mock_cap.release.assert_called_once()
