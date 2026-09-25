"""
Centralised configuration for the PPE Compliance Detection project.

All paths, thresholds, and model parameters are defined here so that
no module needs to hardcode values.  Settings can be overridden via
a YAML file or environment variables in the future.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _project_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class PathSettings:
    """Filesystem paths used across the project."""

    project_root: Path = field(default_factory=_project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def weights_dir(self) -> Path:
        return self.models_dir / "weights"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"


@dataclass
class ModelSettings:
    """YOLOv8 model configuration."""

    model_variant: str = "yolov8n.pt"
    input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"  # "auto", "cpu", "cuda", or "0"


@dataclass
class DetectionSettings:
    """PPE detection class mapping and labels."""

    class_names: List[str] = field(
        default_factory=lambda: [
            "Hardhat",
            "NO-Hardhat",
            "Safety Vest",
            "NO-Safety Vest",
        ]
    )


@dataclass
class ComplianceSettings:
    """Thresholds for the compliance rule engine."""

    required_ppe: List[str] = field(
        default_factory=lambda: ["Hardhat", "Safety Vest"]
    )
    iou_association_threshold: float = 0.3


@dataclass
class Settings:
    """Top-level configuration container."""

    paths: PathSettings = field(default_factory=PathSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    compliance: ComplianceSettings = field(default_factory=ComplianceSettings)

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  project_root={self.paths.project_root},\n"
            f"  model_variant={self.model.model_variant},\n"
            f"  confidence={self.model.confidence_threshold},\n"
            f"  classes={self.detection.class_names}\n"
            f")"
        )
