"""Unit tests for the dataset visualization utility."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from scripts.visualize_dataset import (
    _generate_palette,
    draw_annotations,
    parse_labels,
    yolo_to_xyxy,
)


class TestYoloToXyxy:
    """Tests for YOLO normalised → pixel coordinate conversion."""

    def test_center_box(self) -> None:
        x1, y1, x2, y2 = yolo_to_xyxy(0.5, 0.5, 0.2, 0.2, 640, 480)
        assert x1 == 256  # (0.5 - 0.1) * 640
        assert y1 == 192  # (0.5 - 0.1) * 480
        assert x2 == 384  # (0.5 + 0.1) * 640
        assert y2 == 288  # (0.5 + 0.1) * 480

    def test_clamps_to_image_bounds(self) -> None:
        x1, y1, x2, y2 = yolo_to_xyxy(0.0, 0.0, 0.5, 0.5, 100, 100)
        assert x1 >= 0
        assert y1 >= 0

    def test_full_image_box(self) -> None:
        x1, y1, x2, y2 = yolo_to_xyxy(0.5, 0.5, 1.0, 1.0, 640, 480)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 640
        assert y2 == 480


class TestGeneratePalette:
    """Tests for color palette generation."""

    def test_correct_count(self) -> None:
        palette = _generate_palette(10)
        assert len(palette) == 10

    def test_bgr_tuples(self) -> None:
        palette = _generate_palette(5)
        for color in palette:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_single_color(self) -> None:
        palette = _generate_palette(1)
        assert len(palette) == 1


class TestParseLabels:
    """Tests for tolerant label parsing."""

    def test_valid_labels(self, tmp_path: Path) -> None:
        lbl = tmp_path / "sample.txt"
        lbl.write_text("0 0.5 0.5 0.3 0.4\n5 0.6 0.6 0.2 0.3\n")
        result = parse_labels(lbl, "sample.jpg")
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 5

    def test_empty_label(self, tmp_path: Path) -> None:
        lbl = tmp_path / "empty.txt"
        lbl.write_text("")
        result = parse_labels(lbl, "empty.jpg")
        assert result == []

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        lbl = tmp_path / "bad.txt"
        lbl.write_text("0 0.5\n0 0.5 0.5 0.3 0.4\nabc xyz\n")
        result = parse_labels(lbl, "bad.jpg")
        assert len(result) == 1  # only the valid line

    def test_missing_label_file(self, tmp_path: Path) -> None:
        result = parse_labels(tmp_path / "nonexistent.txt", "missing.jpg")
        assert result == []


class TestDrawAnnotations:
    """Tests for bounding-box drawing."""

    def test_returns_copy(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        original = img.copy()
        annotations = [(0, 0.5, 0.5, 0.3, 0.4)]
        class_names = {0: "Hardhat"}
        palette = _generate_palette(1)

        result = draw_annotations(img, annotations, class_names, palette)

        # Original must be unchanged
        assert np.array_equal(img, original)
        # Result should have drawn something
        assert not np.array_equal(result, original)

    def test_empty_annotations(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_annotations(img, [], {}, _generate_palette(1))
        assert np.array_equal(result, img)

    def test_unknown_class_id(self) -> None:
        """Should not crash when class_id is not in class_names dict."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        annotations = [(99, 0.5, 0.5, 0.2, 0.2)]
        result = draw_annotations(img, annotations, {}, _generate_palette(100))
        assert result.shape == img.shape
