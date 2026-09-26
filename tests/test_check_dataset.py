"""Unit tests for the dataset validation pipeline.

Creates a temporary synthetic dataset structure to exercise every
validation path without needing the real dataset.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from scripts.check_dataset import (
    Issue,
    SplitStats,
    parse_data_yaml,
    parse_label_file,
    resolve_split_dirs,
    validate_dataset,
    save_json_report,
    _report_to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Create a minimal valid YOLOv8 dataset structure."""
    root = tmp_path / "dataset"
    root.mkdir()

    # data.yaml
    yaml_content = textwrap.dedent("""\
        train: ./train/images
        val: ./valid/images
        test: ./test/images
        nc: 10
        names:
          0: Hardhat
          1: Mask
          2: NO-Hardhat
          3: NO-Mask
          4: NO-Safety Vest
          5: Person
          6: Safety Cone
          7: Safety Vest
          8: machinery
          9: vehicle
    """)
    (root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    for split in ["train", "valid", "test"]:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create a valid image (1x1 white PNG)
        _write_tiny_png(img_dir / "sample_001.png")
        (lbl_dir / "sample_001.txt").write_text(
            "0 0.5 0.5 0.3 0.4\n5 0.6 0.6 0.2 0.3\n", encoding="utf-8"
        )

        # Image with no label
        _write_tiny_png(img_dir / "sample_002.png")

        # Label with no image
        (lbl_dir / "sample_003.txt").write_text(
            "7 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )

        # Empty label
        _write_tiny_png(img_dir / "sample_004.png")
        (lbl_dir / "sample_004.txt").write_text("", encoding="utf-8")

    return root


@pytest.fixture
def fake_dataset_bad_labels(tmp_path: Path) -> Path:
    """Dataset with intentionally bad labels."""
    root = tmp_path / "bad_dataset"
    root.mkdir()

    yaml_content = textwrap.dedent("""\
        train: ./train/images
        nc: 5
        names:
          0: Hardhat
          1: NO-Hardhat
          2: NO-Safety Vest
          3: Person
          4: Safety Vest
    """)
    (root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    img_dir = root / "train" / "images"
    lbl_dir = root / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    _write_tiny_png(img_dir / "bad_001.png")
    # Class id out of range, bbox out of range, zero-size bbox
    (lbl_dir / "bad_001.txt").write_text(
        "99 0.5 0.5 0.3 0.4\n"      # class_id out of range
        "0 1.5 0.5 0.3 0.4\n"       # bbox x > 1
        "0 0.5 0.5 0.0 0.4\n"       # bbox width = 0
        "0 0.5 0.5\n"               # too few fields
        "",
        encoding="utf-8",
    )

    return root


def _write_tiny_png(path: Path) -> None:
    """Write a minimal valid 1x1 white PNG file."""
    try:
        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        img.save(path, format="PNG")
    except ImportError:
        # Fallback: write the raw bytes of a minimal valid PNG
        import struct, zlib

        def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            chunk = chunk_type + data
            return (
                struct.pack(">I", len(data))
                + chunk
                + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
            )

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00\xff\xff\xff")
        path.write_bytes(
            sig
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", idat)
            + _png_chunk(b"IEND", b"")
        )


# ---------------------------------------------------------------------------
# Tests: YAML parsing
# ---------------------------------------------------------------------------


class TestParseDataYaml:
    def test_parses_names_dict(self, fake_dataset: Path) -> None:
        data = parse_data_yaml(fake_dataset / "data.yaml")
        assert data["nc"] == 10
        assert data["names"][0] == "Hardhat"
        assert data["names"][5] == "Person"

    def test_parses_paths(self, fake_dataset: Path) -> None:
        data = parse_data_yaml(fake_dataset / "data.yaml")
        assert "train" in data
        assert "val" in data
        assert "test" in data


# ---------------------------------------------------------------------------
# Tests: Split resolution
# ---------------------------------------------------------------------------


class TestResolveSplitDirs:
    def test_finds_all_splits(self, fake_dataset: Path) -> None:
        data = parse_data_yaml(fake_dataset / "data.yaml")
        splits = resolve_split_dirs(fake_dataset, data)
        assert "train" in splits
        assert "valid" in splits
        assert "test" in splits


# ---------------------------------------------------------------------------
# Tests: Label parsing
# ---------------------------------------------------------------------------


class TestParseLabelFile:
    def test_valid_label(self, tmp_path: Path) -> None:
        lbl = tmp_path / "good.txt"
        lbl.write_text("0 0.5 0.5 0.3 0.4\n5 0.6 0.6 0.2 0.3\n")
        issues: list[Issue] = []
        cls_ids, bad_bbox, bad_cls = parse_label_file(lbl, 10, "train", issues)
        assert cls_ids == [0, 5]
        assert bad_bbox == 0
        assert bad_cls == 0
        assert len(issues) == 0

    def test_class_out_of_range(self, tmp_path: Path) -> None:
        lbl = tmp_path / "bad_cls.txt"
        lbl.write_text("99 0.5 0.5 0.3 0.4\n")
        issues: list[Issue] = []
        cls_ids, _, bad_cls = parse_label_file(lbl, 10, "train", issues)
        assert bad_cls == 1
        assert any(i.issue_type == "class_id_out_of_range" for i in issues)

    def test_bbox_out_of_range(self, tmp_path: Path) -> None:
        lbl = tmp_path / "bad_bbox.txt"
        lbl.write_text("0 1.5 0.5 0.3 0.4\n")
        issues: list[Issue] = []
        _, bad_bbox, _ = parse_label_file(lbl, 10, "train", issues)
        assert bad_bbox >= 1

    def test_malformed_line(self, tmp_path: Path) -> None:
        lbl = tmp_path / "short.txt"
        lbl.write_text("0 0.5\n")
        issues: list[Issue] = []
        parse_label_file(lbl, 10, "train", issues)
        assert any(i.issue_type == "malformed_line" for i in issues)


# ---------------------------------------------------------------------------
# Tests: Full validation
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def test_valid_dataset_passes(self, fake_dataset: Path) -> None:
        report = validate_dataset(fake_dataset)
        # Should pass (no critical issues like missing data.yaml)
        assert report.data_yaml_found is True
        assert report.num_classes_yaml == 10
        assert "Hardhat" in report.required_classes_found
        assert "train" in report.splits_found

    def test_counts_images_and_labels(self, fake_dataset: Path) -> None:
        report = validate_dataset(fake_dataset)
        for split in ["train", "valid", "test"]:
            stats = report.split_stats[split]
            assert stats.image_count == 3  # sample_001, 002, 004
            assert stats.label_count == 3  # sample_001, 003, 004

    def test_detects_missing_labels(self, fake_dataset: Path) -> None:
        report = validate_dataset(fake_dataset)
        stats = report.split_stats["train"]
        assert stats.images_without_labels >= 1  # sample_002

    def test_detects_orphan_labels(self, fake_dataset: Path) -> None:
        report = validate_dataset(fake_dataset)
        stats = report.split_stats["train"]
        assert stats.labels_without_images >= 1  # sample_003

    def test_detects_empty_labels(self, fake_dataset: Path) -> None:
        report = validate_dataset(fake_dataset)
        stats = report.split_stats["train"]
        assert stats.empty_labels >= 1  # sample_004

    def test_class_distribution(self, fake_dataset: Path) -> None:
        report = validate_dataset(fake_dataset)
        dist = report.split_stats["train"].class_distribution
        assert "Hardhat" in dist
        assert "Person" in dist

    def test_bad_labels_detected(self, fake_dataset_bad_labels: Path) -> None:
        report = validate_dataset(fake_dataset_bad_labels)
        stats = report.split_stats["train"]
        assert stats.invalid_class_id_count >= 1
        assert stats.invalid_bbox_count >= 1

    def test_missing_data_yaml(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        report = validate_dataset(empty_dir)
        assert report.data_yaml_found is False
        assert report.passed is False


# ---------------------------------------------------------------------------
# Tests: JSON report
# ---------------------------------------------------------------------------


class TestJsonReport:
    def test_json_report_is_valid(self, fake_dataset: Path, tmp_path: Path) -> None:
        report = validate_dataset(fake_dataset)
        json_path = tmp_path / "report.json"
        save_json_report(report, json_path)
        assert json_path.exists()

        with open(json_path) as fh:
            data = json.load(fh)

        assert data["data_yaml_found"] is True
        assert "splits" in data
        assert "issues" in data

    def test_report_to_dict_keys(self, fake_dataset: Path) -> None:
        report = validate_dataset(fake_dataset)
        d = _report_to_dict(report)
        expected_keys = {
            "dataset_path",
            "data_yaml_found",
            "num_classes_yaml",
            "class_names_yaml",
            "required_classes_found",
            "required_classes_missing",
            "splits_found",
            "splits_missing",
            "elapsed_seconds",
            "passed",
            "splits",
            "issues",
        }
        assert expected_keys.issubset(set(d.keys()))
