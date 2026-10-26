"""
Dedicated background monitoring and inference service.

Runs the multi-stage PPE compliance pipeline in a dedicated worker thread,
isolating model execution from ASGI/Uvicorn event loops, exposing structured
person-level PPE compliance status, and broadcasting real-time updates via WebSockets.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

import cv2
import numpy as np
from fastapi import WebSocket

from backend.schemas.monitoring import (
    MonitoringControlResponse,
    MonitoringStatusResponse,
    MonitoringTelemetrySchema,
    TrackedPersonSchema,
)
from detection.pipeline import PipelineResult, PPECompliancePipeline

logger = logging.getLogger("monitoring_service")


class MonitoringService:
    """Orchestrates background video inference, status caching, and WebSocket dispatch."""

    def __init__(
        self,
        pipeline: Optional[PPECompliancePipeline] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.pipeline = pipeline or PPECompliancePipeline()
        self._event_loop = event_loop

        # Threading state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Operational state
        self._is_active: bool = False
        self._source: Optional[Union[int, str]] = None
        self._latest_status: MonitoringStatusResponse = self._empty_status()

        # Connected WebSocket clients
        self._clients: Set[WebSocket] = set()

        # Camera frame orientation (defaults to horizontally flipped / mirrored for natural webcam interaction)
        self._flip_horizontal: bool = True
        self._flip_vertical: bool = False

        # Latest encoded JPEG frame for video streaming
        self._latest_jpeg: Optional[bytes] = None
        self._latest_raw_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Assign the active asyncio event loop for thread-safe WebSocket broadcasts."""
        self._event_loop = loop

    @property
    def is_active(self) -> bool:
        """Return True if background inference thread is running."""
        with self._lock:
            return self._is_active

    @property
    def current_source(self) -> Optional[str]:
        """Return string representation of current input source."""
        with self._lock:
            return str(self._source) if self._source is not None else None

    @property
    def active_tracks_count(self) -> int:
        """Return current number of active tracks in memory."""
        with self._lock:
            return len(self._latest_status.persons)

    def _empty_status(self, source: Optional[str] = None) -> MonitoringStatusResponse:
        """Generate a baseline idle status response."""
        return MonitoringStatusResponse(
            is_active=False,
            source=source,
            total_tracked=0,
            compliant_count=0,
            non_compliant_count=0,
            persons=[],
            telemetry=MonitoringTelemetrySchema(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # -----------------------------------------------------------------------
    # Status Retrieval & Single-Frame Processing
    # -----------------------------------------------------------------------

    def get_current_status(self) -> MonitoringStatusResponse:
        """Fetch the most recent structured PPE compliance status."""
        with self._lock:
            return self._latest_status

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Fetch the most recent annotated JPEG frame bytes for video streaming."""
        with self._frame_lock:
            return self._latest_jpeg

    def get_latest_raw_frame_jpeg(self) -> Optional[bytes]:
        """Fetch the most recent unannotated raw camera frame as JPEG bytes."""
        with self._frame_lock:
            return self._latest_raw_jpeg

    def process_single_frame(
        self,
        frame: np.ndarray,
        source_label: str = "manual_frame",
    ) -> MonitoringStatusResponse:
        """Process a single frame synchronously and update latest status.

        Allows evaluating static images or API-submitted frames without running
        a continuous background video loop.
        """
        if frame is None or frame.size == 0:
            return self._empty_status(source=source_label)

        result: PipelineResult = self.pipeline.process_frame(frame, draw_overlay=False)
        status = self._build_status_response(result, is_active=self.is_active, source=source_label)

        with self._lock:
            self._latest_status = status

        self._schedule_broadcast(status)
        return status

    # -----------------------------------------------------------------------
    # Background Inference Worker Lifecycle
    # -----------------------------------------------------------------------

    def start(
        self,
        source: Union[int, str] = 0,
        flip_horizontal: Optional[bool] = None,
        flip_vertical: Optional[bool] = None,
    ) -> MonitoringControlResponse:
        """Launch inference in a dedicated background worker thread."""
        with self._lock:
            if flip_horizontal is not None:
                self._flip_horizontal = flip_horizontal
            if flip_vertical is not None:
                self._flip_vertical = flip_vertical

            if self._is_active and self._thread is not None and self._thread.is_alive():
                return MonitoringControlResponse(
                    status="running",
                    source=str(self._source),
                    message=f"Monitoring already active on source: {self._source}",
                )

            self._stop_event.clear()
            self._source = source
            self._is_active = True

            self._thread = threading.Thread(
                target=self._worker_loop,
                args=(source,),
                name="PPEInferenceWorker",
                daemon=True,
            )
            self._thread.start()

        logger.info("Started background monitoring worker on source: %s (flip_h=%s, flip_v=%s)", source, self._flip_horizontal, self._flip_vertical)
        return MonitoringControlResponse(
            status="running",
            source=str(source),
            message=f"Monitoring started on source: {source}",
        )

    def set_flip(
        self,
        flip_horizontal: Optional[bool] = None,
        flip_vertical: Optional[bool] = None,
    ) -> Dict[str, bool]:
        """Update camera orientation flip settings dynamically."""
        with self._lock:
            if flip_horizontal is not None:
                self._flip_horizontal = flip_horizontal
            if flip_vertical is not None:
                self._flip_vertical = flip_vertical
            return {
                "flip_horizontal": self._flip_horizontal,
                "flip_vertical": self._flip_vertical,
            }

    def stop(self) -> MonitoringControlResponse:
        """Signal background worker to terminate and wait for clean shutdown."""
        with self._lock:
            if not self._is_active:
                return MonitoringControlResponse(
                    status="stopped",
                    source=None,
                    message="Monitoring is already stopped.",
                )

            self._stop_event.set()
            worker_thread = self._thread

        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=3.0)

        with self._lock:
            self._is_active = False
            prev_source = self._source
            self._source = None
            self._latest_status = self._empty_status(source=None)

        with self._frame_lock:
            self._latest_jpeg = None
            self._latest_raw_jpeg = None

        # Notify WebSockets of stopped state
        self._schedule_broadcast(self._latest_status)
        logger.info("Stopped background monitoring worker.")

        return MonitoringControlResponse(
            status="stopped",
            source=str(prev_source) if prev_source is not None else None,
            message="Monitoring stopped successfully.",
        )

    # -----------------------------------------------------------------------
    # Worker Loop
    # -----------------------------------------------------------------------

    def _worker_loop(self, source: Union[int, str]) -> None:
        """Target for dedicated background thread. Continually captures and evaluates frames."""
        camera_source: Union[int, str] = source
        if isinstance(source, str) and source.isdigit():
            camera_source = int(source)

        is_simulation = False
        fallback_img_path = _PROJECT_ROOT / "outputs" / "pipeline_test.jpg"

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            if fallback_img_path.exists():
                logger.warning(
                    "Physical camera source %s not available (headless cloud environment). "
                    "Running live monitoring loop using simulation test frame: %s",
                    source,
                    fallback_img_path.name,
                )
                is_simulation = True
            else:
                logger.error("Could not open video source: %s", source)
                with self._lock:
                    self._is_active = False
                    self._latest_status = self._empty_status(source=str(source))
                return

        # Optimization: Minimize capture latency and match inference resolution
        if not is_simulation and isinstance(camera_source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        logger.info("Inference worker running on source: %s (simulation=%s)", source, is_simulation)

        sim_frame = cv2.imread(str(fallback_img_path)) if is_simulation else None

        try:
            while not self._stop_event.is_set():
                if is_simulation:
                    time.sleep(1.0 / 25.0)  # Simulate 25 FPS stream rate on cloud
                    frame = sim_frame.copy() if sim_frame is not None else None
                    if frame is None:
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # If reading a file, loop back or terminate
                        if isinstance(source, str) and not source.isdigit():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        # For camera, wait briefly before retrying
                        time.sleep(0.05)
                        continue

                # Apply camera flip/mirror orientation if configured
                with self._lock:
                    flip_h = self._flip_horizontal
                    flip_v = self._flip_vertical

                if flip_h and flip_v:
                    frame = cv2.flip(frame, -1)
                elif flip_h:
                    frame = cv2.flip(frame, 1)
                elif flip_v:
                    frame = cv2.flip(frame, 0)

                # Encode raw frame (before annotations) for clean face enrollment snapshots
                ret_raw, raw_buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                # Run multi-stage pipeline with visual overlay enabled
                result: PipelineResult = self.pipeline.process_frame(frame, draw_overlay=True)

                display_frame = result.annotated_frame if result.annotated_frame is not None else frame
                ret_enc, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                with self._frame_lock:
                    if ret_enc:
                        self._latest_jpeg = buffer.tobytes()
                    if ret_raw:
                        self._latest_raw_jpeg = raw_buf.tobytes()

                status = self._build_status_response(
                    result, is_active=True, source=str(source)
                )

                with self._lock:
                    self._latest_status = status

                # Broadcast structured telemetry & compliance state
                self._schedule_broadcast(status)

                # Regulate loop rate (~30 FPS max)
                time.sleep(0.01)

        except Exception as exc:
            logger.exception("Error in inference worker loop: %s", exc)
        finally:
            cap.release()
            with self._frame_lock:
                self._latest_jpeg = None
            with self._lock:
                self._is_active = False
            logger.info("Video capture released for source: %s", source)

    # -----------------------------------------------------------------------
    # Response Formatting
    # -----------------------------------------------------------------------

    def _build_status_response(
        self,
        result: PipelineResult,
        is_active: bool,
        source: Optional[str],
    ) -> MonitoringStatusResponse:
        """Convert a PipelineResult into a structured Pydantic MonitoringStatusResponse."""
        person_schemas: List[TrackedPersonSchema] = []

        for rec in result.records:
            person_schemas.append(
                TrackedPersonSchema(
                    track_id=rec.track_id,
                    employee_id=rec.employee_id,
                    employee_name=rec.employee_name,
                    hardhat=rec.hardhat,
                    safety_vest=rec.safety_vest,
                    compliance_status=rec.compliance_status,
                    missing_items=list(rec.missing_items),
                    detection_confidence=rec.detection_confidence,
                    bbox=list(rec.bbox),
                    is_compliant=rec.is_compliant,
                    face_similarity=rec.face_similarity,
                )
            )

        return MonitoringStatusResponse(
            is_active=is_active,
            source=source,
            total_tracked=result.total_tracked,
            compliant_count=result.compliant_count,
            non_compliant_count=result.non_compliant_count,
            persons=person_schemas,
            telemetry=MonitoringTelemetrySchema(
                fps=result.telemetry.fps,
                total_ms=result.telemetry.total_ms,
                detection_ms=result.telemetry.detection_ms,
                association_ms=result.telemetry.association_ms,
                recognition_ms=result.telemetry.recognition_ms,
                smoothing_ms=result.telemetry.smoothing_ms,
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # -----------------------------------------------------------------------
    # WebSocket Dispatch
    # -----------------------------------------------------------------------

    async def connect_client(self, websocket: WebSocket) -> None:
        """Register a new WebSocket client and send initial state snapshot."""
        await websocket.accept()
        self._clients.add(websocket)
        logger.info("WebSocket client connected. Active clients: %d", len(self._clients))

        # Send immediate initial status snapshot
        status = self.get_current_status()
        initial_msg = {"type": "snapshot", "data": status.model_dump()}
        try:
            await websocket.send_json(initial_msg)
        except Exception:
            self._clients.discard(websocket)

    def disconnect_client(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket client."""
        self._clients.discard(websocket)
        logger.info("WebSocket client disconnected. Remaining clients: %d", len(self._clients))

    async def broadcast_status(self, status: MonitoringStatusResponse) -> None:
        """Transmit structured monitoring data to all active WebSocket connections.

        Strictly does NOT include raw video frames.
        """
        if not self._clients:
            return

        payload = {"type": "monitoring_update", "data": status.model_dump()}
        stale_clients: List[WebSocket] = []

        for client in list(self._clients):
            try:
                await client.send_json(payload)
            except Exception:
                stale_clients.append(client)

        for client in stale_clients:
            self._clients.discard(client)

    def _schedule_broadcast(self, status: MonitoringStatusResponse) -> None:
        """Dispatch broadcast from the inference worker thread to the async event loop."""
        if not self._clients or self._event_loop is None:
            return

        try:
            if not self._event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.broadcast_status(status), self._event_loop
                )
        except RuntimeError:
            pass

    def shutdown(self) -> None:
        """Cleanup handler on application shutdown."""
        self.stop()
