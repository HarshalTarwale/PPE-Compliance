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
    MonitoringFlipRequest,
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
    return service.start(
        source=payload.source,
        flip_horizontal=payload.flip_horizontal,
        flip_vertical=payload.flip_vertical,
    )


@router.post(
    "/monitoring/flip",
    summary="Dynamically toggle camera flip / mirror orientation",
)
async def toggle_camera_flip(
    payload: MonitoringFlipRequest,
    request: Request,
):
    """Toggle horizontal or vertical camera flip dynamically."""
    service = get_monitoring_service(request)
    return service.set_flip(
        flip_horizontal=payload.flip_horizontal,
        flip_vertical=payload.flip_vertical,
    )


@router.post(
    "/monitoring/stop",
    response_model=MonitoringControlResponse,
    summary="Stop background stream monitoring",
)
async def stop_monitoring(request: Request) -> MonitoringControlResponse:
    """Stop the background inference worker loop."""
    service = get_monitoring_service(request)
    return service.stop()


@router.get(
    "/monitoring/snapshot",
    summary="Get single JPEG snapshot of camera frame (clean unannotated for employee enrollment)",
)
async def get_camera_snapshot(request: Request, raw: bool = True):
    """Retrieve the most recent camera frame as an unannotated JPEG image for face registration."""
    from fastapi.responses import Response
    service = get_monitoring_service(request)
    jpeg_bytes = service.get_latest_raw_frame_jpeg() if raw else service.get_latest_frame_jpeg()
    if jpeg_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No camera frame available. Please ensure stream is active or connect webcam.",
        )
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get(
    "/monitoring/video_feed",
    summary="Real-Time Annotated Camera MJPEG Stream",
)
async def get_video_feed(request: Request):
    """Streams live camera frames as MJPEG multipart stream for in-browser visual inspection."""
    import asyncio
    from fastapi.responses import StreamingResponse

    service = get_monitoring_service(request)

    async def frame_stream():
        while True:
            if await request.is_disconnected():
                break
            jpeg_bytes = service.get_latest_frame_jpeg()
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
