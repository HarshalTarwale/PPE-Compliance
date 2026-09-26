"""Unit tests for the model evaluation pipeline (training/evaluate.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from training.evaluate import (
    TARGET_PPE_CLASSES,
    build_parser,
    extract_evaluation_metrics,
    generate_text_report,
    validate_data_path,
    validate_model_path,
)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidatePaths:
    """Test path validation functions."""

    def test_validate_model_path_exists(self, tmp_path: Path):
        fake_model = tmp_path / "best.pt"
        fake_model.write_text("weights")
        result = validate_model_path(fake_model)
        assert result == fake_model

    def test_validate_model_path_missing(self, tmp_path: Path):
        fake_model = tmp_path / "nonexistent.pt"
        with pytest.raises(SystemExit) as exc_info:
            validate_model_path(fake_model)
        assert exc_info.value.code == 1

    def test_validate_model_path_is_dir(self, tmp_path: Path):
        fake_dir = tmp_path / "model_dir"
        fake_dir.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            validate_model_path(fake_dir)
        assert exc_info.value.code == 1

    def test_validate_data_path_file(self, tmp_path: Path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names: [a, b]\nnc: 2\n")
        assert validate_data_path(data_yaml) == data_yaml

    def test_validate_data_path_dir(self, tmp_path: Path):
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        data_yaml = data_dir / "data.yaml"
        data_yaml.write_text("names: [a, b]\nnc: 2\n")
        assert validate_data_path(data_dir) == data_yaml

    def test_validate_data_path_missing(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            validate_data_path(empty_dir)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Metric extraction tests
# ---------------------------------------------------------------------------


class TestExtractEvaluationMetrics:
    """Test extraction of metrics from DetMetrics."""

    @pytest.fixture
    def mock_metrics(self):
        """Create a mock DetMetrics object with realistic values."""
        m = MagicMock()
        # Overall
        m.box.mp = 0.825
        m.box.mr = 0.650
        m.box.map50 = 0.742
        m.box.map = 0.518

        # Classes: 0: Hardhat, 1: NO-Hardhat, 2: Person, 3: Safety Vest, 4: NO-Safety Vest
        m.box.ap_class_index = np.array([0, 1, 2, 3, 4])
        m.box.p = np.array([0.87, 0.70, 0.83, 0.85, 0.82])
        m.box.r = np.array([0.61, 0.51, 0.65, 0.63, 0.58])
        m.box.ap50 = np.array([0.74, 0.54, 0.77, 0.72, 0.64])
        m.box.ap = np.array([0.51, 0.28, 0.56, 0.47, 0.35])

        m.speed = {
            "preprocess": 1.5,
            "inference": 3.2,
            "loss": 0.0,
            "postprocess": 2.1,
        }
        m.nt_per_class = np.array([79, 69, 166, 41, 106])
        return m

    def test_overall_metrics_extraction(self, mock_metrics):
        names = {
            0: "Hardhat",
            1: "NO-Hardhat",
            2: "Person",
            3: "Safety Vest",
            4: "NO-Safety Vest",
        }
        results = extract_evaluation_metrics(mock_metrics, names)

        assert results["overall"]["precision"] == 0.825
        assert results["overall"]["recall"] == 0.65
        assert results["overall"]["map50"] == 0.742
        assert results["overall"]["map50_95"] == 0.518

    def test_target_classes_extraction(self, mock_metrics):
        names = {
            0: "Hardhat",
            1: "NO-Hardhat",
            2: "Person",
            3: "Safety Vest",
            4: "NO-Safety Vest",
        }
        results = extract_evaluation_metrics(mock_metrics, names)

        for target in TARGET_PPE_CLASSES:
            assert target in results["target_classes"]
            entry = results["target_classes"][target]
            assert "precision" in entry
            assert "recall" in entry
            assert "map50" in entry
            assert "map50_95" in entry
            assert "instances" in entry
            assert entry["instances"] > 0

    def test_speed_metrics_calculation(self, mock_metrics):
        names = {0: "Hardhat"}
        results = extract_evaluation_metrics(mock_metrics, names)
        speed = results["speed"]

        assert speed["preprocess_ms"] == 1.5
        assert speed["inference_ms"] == 3.2
        assert speed["postprocess_ms"] == 2.1
        assert speed["total_latency_ms"] == 6.8
        assert speed["fps"] == round(1000.0 / 6.8, 1)


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateTextReport:
    """Test human-readable report formatting."""

    def test_report_contains_all_sections(self):
        sample_data = {
            "timestamp": "2026-09-25T22:00:00",
            "model_path": "models/weights/best.pt",
            "data_yaml": "data/data.yaml",
            "split": "val",
            "device": "0",
            "overall": {
                "precision": 0.827,
                "recall": 0.651,
                "map50": 0.772,
                "map50_95": 0.559,
            },
            "speed": {
                "preprocess_ms": 1.5,
                "inference_ms": 3.2,
                "postprocess_ms": 2.1,
                "total_latency_ms": 6.8,
                "fps": 147.1,
            },
            "target_classes": {
                "Person": {
                    "class_name": "Person",
                    "instances": 166,
                    "precision": 0.827,
                    "recall": 0.651,
                    "map50": 0.772,
                    "map50_95": 0.559,
                },
                "Hardhat": {
                    "class_name": "Hardhat",
                    "instances": 79,
                    "precision": 0.873,
                    "recall": 0.607,
                    "map50": 0.738,
                    "map50_95": 0.514,
                },
                "NO-Hardhat": {
                    "class_name": "NO-Hardhat",
                    "instances": 69,
                    "precision": 0.699,
                    "recall": 0.507,
                    "map50": 0.543,
                    "map50_95": 0.282,
                },
                "Safety Vest": {
                    "class_name": "Safety Vest",
                    "instances": 41,
                    "precision": 0.855,
                    "recall": 0.634,
                    "map50": 0.719,
                    "map50_95": 0.470,
                },
                "NO-Safety Vest": {
                    "class_name": "NO-Safety Vest",
                    "instances": 106,
                    "precision": 0.828,
                    "recall": 0.575,
                    "map50": 0.639,
                    "map50_95": 0.351,
                },
            },
            "per_class": {},
        }

        report = generate_text_report(sample_data)

        # Check titles & sections
        assert "MODEL EVALUATION REPORT" in report
        assert "OVERALL EVALUATION METRICS" in report
        assert "TARGET PPE COMPLIANCE CLASSES" in report
        assert "SPEED & LATENCY BENCHMARK" in report

        # Check target class rows exist
        for target in TARGET_PPE_CLASSES:
            assert target in report


# ---------------------------------------------------------------------------
# CLI Argument tests
# ---------------------------------------------------------------------------


class TestCLIParser:
    """Test command-line argument parser."""

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--data", "data/data.yaml"])
        assert args.data == "data/data.yaml"
        assert args.model == "models/weights/best.pt"
        assert args.split == "val"
        assert args.device == "auto"
        assert args.output_dir == "outputs/evaluation"

    def test_parser_custom_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "custom.pt",
                "--data",
                "custom_data.yaml",
                "--split",
                "test",
                "--device",
                "0",
                "--output-dir",
                "custom/out",
            ]
        )
        assert args.model == "custom.pt"
        assert args.data == "custom_data.yaml"
        assert args.split == "test"
        assert args.device == "0"
        assert args.output_dir == "custom/out"
