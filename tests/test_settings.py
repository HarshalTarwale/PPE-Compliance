"""Unit tests for config.settings module."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    ComplianceSettings,
    DetectionSettings,
    ModelSettings,
    PathSettings,
    Settings,
)


class TestPathSettings:
    """Tests for PathSettings dataclass."""

    def test_project_root_exists(self) -> None:
        paths = PathSettings()
        assert paths.project_root.exists()

    def test_data_dir_is_under_root(self) -> None:
        paths = PathSettings()
        assert paths.data_dir == paths.project_root / "data"

    def test_weights_dir_is_under_models(self) -> None:
        paths = PathSettings()
        assert paths.weights_dir == paths.models_dir / "weights"


class TestModelSettings:
    """Tests for ModelSettings defaults."""

    def test_default_model_variant(self) -> None:
        model = ModelSettings()
        assert model.model_variant == "yolov8n.pt"

    def test_default_confidence(self) -> None:
        model = ModelSettings()
        assert 0.0 < model.confidence_threshold < 1.0

    def test_default_iou(self) -> None:
        model = ModelSettings()
        assert 0.0 < model.iou_threshold < 1.0


class TestDetectionSettings:
    """Tests for DetectionSettings defaults."""

    def test_class_count(self) -> None:
        det = DetectionSettings()
        assert len(det.class_names) == 4

    def test_expected_classes(self) -> None:
        det = DetectionSettings()
        assert "Hardhat" in det.class_names
        assert "NO-Hardhat" in det.class_names
        assert "Safety Vest" in det.class_names
        assert "NO-Safety Vest" in det.class_names


class TestComplianceSettings:
    """Tests for ComplianceSettings defaults."""

    def test_required_ppe(self) -> None:
        comp = ComplianceSettings()
        assert "Hardhat" in comp.required_ppe
        assert "Safety Vest" in comp.required_ppe


class TestSettings:
    """Tests for the top-level Settings container."""

    def test_settings_has_all_subsections(self) -> None:
        settings = Settings()
        assert isinstance(settings.paths, PathSettings)
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.detection, DetectionSettings)
        assert isinstance(settings.compliance, ComplianceSettings)

    def test_repr(self) -> None:
        settings = Settings()
        text = repr(settings)
        assert "Settings(" in text
        assert "yolov8n.pt" in text
