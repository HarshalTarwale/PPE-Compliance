"""Detection package — YOLOv8 PPE detection, inference, and association logic."""

from detection.detector import Detection, DetectionResult, PPEDetector
from detection.ppe_association import (
    AssociatedPerson,
    AssociationConfig,
    AssociationResult,
    PPEAssociator,
)
from detection.pipeline import (
    ActiveTrack,
    PipelineResult,
    PipelineTelemetry,
    PPECompliancePipeline,
    SimpleIoUTracker,
    TrackedPersonRecord,
)
from detection.webcam import run_webcam

__all__ = [
    "ActiveTrack",
    "AssociatedPerson",
    "AssociationConfig",
    "AssociationResult",
    "Detection",
    "DetectionResult",
    "PPEAssociator",
    "PPECompliancePipeline",
    "PPEDetector",
    "PipelineResult",
    "PipelineTelemetry",
    "SimpleIoUTracker",
    "TrackedPersonRecord",
    "run_webcam",
]
