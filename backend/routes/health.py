"""System health and diagnostic endpoints."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from backend.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System Health & Diagnostics")
async def get_health(request: Request) -> HealthResponse:
    """Check health, operational status, compute device, and model readiness."""
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = max(0.0, time.time() - start_time)

    monitoring_service = getattr(request.app.state, "monitoring_service", None)
    is_active = monitoring_service.is_active if monitoring_service else False
    active_tracks = monitoring_service.active_tracks_count if monitoring_service else 0

    # Model readiness checks
    detector_loaded = False
    inference_device = "unknown"
    if monitoring_service and monitoring_service.pipeline and monitoring_service.pipeline.detector:
        detector_loaded = monitoring_service.pipeline.detector.model is not None
        inference_device = str(monitoring_service.pipeline.detector.device)

    # Face models check
    face_models_loaded = False
    if monitoring_service and monitoring_service.pipeline and monitoring_service.pipeline.face_service:
        engine = monitoring_service.pipeline.face_service.engine
        face_models_loaded = (
            engine is not None and engine.yunet_model is not None and engine.sface_model is not None
        )

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=round(uptime, 2),
        detector_loaded=detector_loaded,
        face_models_loaded=face_models_loaded,
        inference_device=inference_device,
        monitoring_active=is_active,
        active_tracks=active_tracks,
    )
