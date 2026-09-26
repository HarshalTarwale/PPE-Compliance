"""Health check schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for GET /health endpoint."""

    status: str = Field("healthy", description="Overall service status ('healthy', 'degraded')")
    version: str = Field("1.0.0", description="API version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current UTC timestamp in ISO 8601 format",
    )
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    detector_loaded: bool = Field(..., description="Whether YOLO detector model is loaded and ready")
    face_models_loaded: bool = Field(..., description="Whether YuNet and SFace models are loaded")
    inference_device: str = Field(..., description="Active compute device (e.g. 'cuda:0' or 'cpu')")
    monitoring_active: bool = Field(..., description="Whether real-time video stream monitoring is running")
    active_tracks: int = Field(0, description="Current number of active tracked persons in memory")
