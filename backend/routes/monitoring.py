"""Real-time monitoring and WebSocket streaming endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from backend.schemas.monitoring import (
    MonitoringControlRequest,
    MonitoringControlResponse,
    MonitoringStatusResponse,
)
from backend.services.monitoring_service import MonitoringService

logger = logging.getLogger("monitoring_routes")
router = APIRouter(tags=["Monitoring"])


def get_monitoring_service(request: Request) -> MonitoringService:
    """Retrieve MonitoringService from application state."""
    service: Optional[MonitoringService] = getattr(request.app.state, "monitoring_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring service is not initialized.",
        )
    return service


@router.get(
    "/monitoring/status",
    response_model=MonitoringStatusResponse,
    summary="Get current monitoring status snapshot",
)
async def get_monitoring_status(request: Request) -> MonitoringStatusResponse:
    """Retrieve current structured PPE compliance status for all tracked individuals."""
    service = get_monitoring_service(request)
    return service.get_current_status()


@router.post(
    "/monitoring/start",
    response_model=MonitoringControlResponse,
    summary="Start background stream monitoring",
)
async def start_monitoring(
    payload: MonitoringControlRequest,
    request: Request,
) -> MonitoringControlResponse:
    """Start background video capture and PPE inference loop on a camera or video file."""
    service = get_monitoring_service(request)
    return service.start(source=payload.source)


@router.post(
    "/monitoring/stop",
    response_model=MonitoringControlResponse,
    summary="Stop background stream monitoring",
)
async def stop_monitoring(request: Request) -> MonitoringControlResponse:
    """Stop the background inference worker loop."""
    service = get_monitoring_service(request)
    return service.stop()


# ---------------------------------------------------------------------------
# WebSocket Support
# ---------------------------------------------------------------------------


async def _handle_websocket(websocket: WebSocket) -> None:
    """Internal handler for real-time WebSocket connection."""
    monitoring_service: Optional[MonitoringService] = getattr(
        websocket.app.state, "monitoring_service", None
    )

    if monitoring_service is None:
        await websocket.close(code=1011, reason="Monitoring service unavailable")
        return

    await monitoring_service.connect_client(websocket)

    try:
        while True:
            # Keep connection alive; clients may send heartbeats/ping
            data = await websocket.receive_text()
            if data.strip().lower() == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket client disconnected with exception: %s", exc)
    finally:
        monitoring_service.disconnect_client(websocket)


@router.websocket("/monitoring/ws")
async def websocket_monitoring_primary(websocket: WebSocket) -> None:
    """WebSocket stream for real-time structured PPE telemetry and compliance data."""
    await _handle_websocket(websocket)


@router.websocket("/ws/monitoring")
async def websocket_monitoring_alias(websocket: WebSocket) -> None:
    """Convenience alias for the monitoring WebSocket stream."""
    await _handle_websocket(websocket)
