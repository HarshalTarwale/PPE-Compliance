"""Monitoring and real-time PPE compliance schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TrackedPersonSchema(BaseModel):
    """Structured PPE and identity record for a single tracked person."""

    track_id: str = Field(..., description="Unique track ID in current stream (e.g. 'track_1')")
    employee_id: str = Field("UNKNOWN", description="Matched employee ID or 'UNKNOWN'")
    employee_name: str = Field("UNKNOWN", description="Matched employee name or 'UNKNOWN'")
    hardhat: bool = Field(..., description="Whether hardhat is present")
    safety_vest: bool = Field(..., description="Whether safety vest is present")
    compliance_status: str = Field(..., description="'COMPLIANT' or 'NON_COMPLIANT'")
    missing_items: List[str] = Field(default_factory=list, description="List of missing safety equipment")
    detection_confidence: float = Field(..., description="Confidence score for person detection (0.0 - 1.0)")
    bbox: List[int] = Field(default_factory=list, description="Bounding box [x1, y1, x2, y2]")
    is_compliant: bool = Field(..., description="Boolean compliance indicator")
    face_similarity: Optional[float] = Field(None, description="Cosine similarity score for verified face")


class SafetyEventSchema(BaseModel):
    """A logged safety event: a PPE violation or an unknown person in view."""

    id: str = Field(..., description="Unique event ID")
    type: str = Field(..., description="'ppe_violation' or 'unknown_person'")
    timestamp: str = Field(..., description="ISO 8601 UTC time the event was recorded")
    track_id: str = Field(..., description="Tracker ID of the person involved")
    employee_id: str = Field("UNKNOWN", description="Matched employee ID or 'UNKNOWN'")
    employee_name: str = Field("UNKNOWN", description="Matched employee name or 'UNKNOWN'")
    missing_items: List[str] = Field(default_factory=list, description="Missing PPE at the time")
    message: str = Field("", description="Human-readable summary")
    source: Optional[str] = Field(None, description="Camera source the event came from")


class SafetyEventListResponse(BaseModel):
    """Schema for GET /monitoring/events response."""

    total: int = Field(..., description="Total number of events stored")
    events: List[SafetyEventSchema] = Field(default_factory=list, description="Events, newest first")


class MonitoringTelemetrySchema(BaseModel):
    """Pipeline latency and performance metrics."""

    fps: float = Field(0.0, description="Frames processed per second")
    total_ms: float = Field(0.0, description="Total pipeline latency in ms")
    detection_ms: float = Field(0.0, description="Object detection inference latency in ms")
    association_ms: float = Field(0.0, description="PPE association latency in ms")
    recognition_ms: float = Field(0.0, description="Face recognition latency in ms")
    smoothing_ms: float = Field(0.0, description="Temporal smoothing latency in ms")


class MonitoringStatusResponse(BaseModel):
    """Schema for GET /monitoring/status response."""

    is_active: bool = Field(..., description="Whether real-time inference loop is currently running")
    source: Optional[str] = Field(None, description="Active camera index or video source path")
    total_tracked: int = Field(0, description="Total tracked persons currently in the scene")
    compliant_count: int = Field(0, description="Number of compliant workers")
    non_compliant_count: int = Field(0, description="Number of workers violating PPE policies")
    persons: List[TrackedPersonSchema] = Field(
        default_factory=list, description="Structured PPE status per person"
    )
    telemetry: MonitoringTelemetrySchema = Field(
        default_factory=MonitoringTelemetrySchema, description="Timing and FPS telemetry"
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of status evaluation")
    events: List[SafetyEventSchema] = Field(
        default_factory=list, description="Safety events raised by this frame (new alerts)"
    )


class MonitoringControlRequest(BaseModel):
    """Schema for POST /monitoring/start control request."""

    source: Union[int, str] = Field(0, description="Camera index (e.g. 0) or path to video file")
    flip_horizontal: bool = Field(True, description="Flip camera horizontally (mirror mode)")
    flip_vertical: bool = Field(False, description="Flip camera vertically")


class MonitoringFlipRequest(BaseModel):
    """Schema for POST /monitoring/flip control request."""

    flip_horizontal: Optional[bool] = Field(None, description="Flip camera horizontally (mirror mode)")
    flip_vertical: Optional[bool] = Field(None, description="Flip camera vertically")


class MonitoringControlResponse(BaseModel):
    """Schema for POST /monitoring/start and POST /monitoring/stop control response."""

    status: str = Field(..., description="'running', 'stopped', or 'error'")
    source: Optional[str] = Field(None, description="Stream source identifier")
    message: str = Field(..., description="Status description")
