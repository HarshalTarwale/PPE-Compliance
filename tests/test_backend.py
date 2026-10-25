"""
Unit and integration tests for FastAPI backend (backend/main.py, routes, services).

Tests:
  1. GET /health (diagnostics, uptime, model status, active tracks)
  2. GET /employees (empty list, populated list, schema validation)
  3. POST /employees (success 201, empty fields 400, rejected faces 400)
  4. GET /employees/{id} (found 200, missing 404)
  5. DELETE /employees/{id} (deleted 200, missing 404)
  6. GET /monitoring/status (structured PPE status format, compliance numbers)
  7. POST /monitoring/start & POST /monitoring/stop (lifecycle control)
  8. WebSocket /ws/monitoring and /monitoring/ws (connection, initial snapshot, structured updates, ping-pong)
  9. CORS headers verification
"""

from __future__ import annotations

import io
from typing import List, Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.schemas.employee import (
    EmployeeListResponse,
    EmployeeRegistrationResponse,
    EmployeeResponse,
)
from backend.schemas.monitoring import (
    MonitoringControlResponse,
    MonitoringStatusResponse,
    MonitoringTelemetrySchema,
    TrackedPersonSchema,
)
from backend.services.employee_service import EmployeeService
from backend.services.monitoring_service import MonitoringService
from detection.pipeline import PipelineResult, TrackedPersonRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_test_image_bytes() -> bytes:
    """Create a 100x100 white PNG in bytes."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


@pytest.fixture
def mock_employee_service() -> MagicMock:
    """Mock EmployeeService with standard fixture data."""
    service = MagicMock(spec=EmployeeService)

    sample_emp = EmployeeResponse(
        employee_id="EMP001",
        employee_name="Jane Doe",
        registered_at="2026-09-25T12:00:00Z",
        sample_count=3,
        embedding_dim=128,
        metadata={"department": "Engineering"},
    )

    service.list_employees.return_value = EmployeeListResponse(
        total=1,
        employees=[sample_emp],
    )

    def _get_employee(emp_id: str) -> Optional[EmployeeResponse]:
        if emp_id == "EMP001":
            return sample_emp
        return None

    service.get_employee.side_effect = _get_employee

    def _delete_employee(emp_id: str) -> bool:
        return emp_id == "EMP001"

    service.delete_employee.side_effect = _delete_employee

    def _register_employee(
        employee_id: str,
        employee_name: str,
        image_bytes_list: List[bytes],
        allow_overwrite: bool = False,
        **kwargs,
    ):
        if not employee_id or not employee_name:
            return EmployeeRegistrationResponse(
                success=False,
                employee_id=employee_id,
                employee_name=employee_name,
                valid_samples=0,
                rejected_samples=len(image_bytes_list),
                message="ID and name are required.",
            )
        return EmployeeRegistrationResponse(
            success=True,
            employee_id=employee_id,
            employee_name=employee_name,
            valid_samples=len(image_bytes_list),
            rejected_samples=0,
            message="Successfully registered employee.",
        )

    service.register_employee.side_effect = _register_employee
    return service


@pytest.fixture
def mock_monitoring_service() -> MagicMock:
    """Mock MonitoringService providing structured status."""
    service = MagicMock(spec=MonitoringService)
    service.is_active = False
    service.active_tracks_count = 1
    service.current_source = None

    sample_person = TrackedPersonSchema(
        track_id="track_1",
        employee_id="EMP001",
        employee_name="Jane Doe",
        hardhat=True,
        safety_vest=True,
        compliance_status="COMPLIANT",
        missing_items=[],
        detection_confidence=0.92,
        bbox=[100, 50, 300, 450],
        is_compliant=True,
        face_similarity=0.74,
    )

    sample_status = MonitoringStatusResponse(
        is_active=False,
        source=None,
        total_tracked=1,
        compliant_count=1,
        non_compliant_count=0,
        persons=[sample_person],
        telemetry=MonitoringTelemetrySchema(
            fps=32.5,
            total_ms=30.8,
            detection_ms=12.2,
            association_ms=2.1,
            recognition_ms=15.0,
            smoothing_ms=1.5,
        ),
        timestamp="2026-09-25T12:00:00Z",
    )

    service.get_current_status.return_value = sample_status

    service.start.return_value = MonitoringControlResponse(
        status="running",
        source="0",
        message="Monitoring started on source: 0",
    )

    service.stop.return_value = MonitoringControlResponse(
        status="stopped",
        source=None,
        message="Monitoring stopped successfully.",
    )

    service.set_flip.return_value = {
        "flip_horizontal": True,
        "flip_vertical": False,
    }

    # Fake pipeline attributes for /health
    pipeline_mock = MagicMock()
    pipeline_mock.detector.model = "yolov8n_loaded"
    pipeline_mock.detector.device = "cpu"
    pipeline_mock.face_service.engine.yunet_model = "yunet_loaded"
    pipeline_mock.face_service.engine.sface_model = "sface_loaded"
    service.pipeline = pipeline_mock

    # Connect client simulation for WebSockets
    async def _connect_client(ws):
        await ws.accept()
        await ws.send_json({"type": "snapshot", "data": sample_status.model_dump()})

    service.connect_client = _connect_client
    service.disconnect_client = MagicMock()

    return service


@pytest.fixture
def client(mock_employee_service: MagicMock, mock_monitoring_service: MagicMock) -> TestClient:
    """TestClient configured with mock services."""
    app = create_app(
        employee_service=mock_employee_service,
        monitoring_service=mock_monitoring_service,
    )
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_get_health_status(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0.0
        assert data["detector_loaded"] is True
        assert data["face_models_loaded"] is True
        assert data["inference_device"] == "cpu"
        assert data["monitoring_active"] is False
        assert data["active_tracks"] == 1


# ---------------------------------------------------------------------------
# Employee Management Tests
# ---------------------------------------------------------------------------


class TestEmployeeEndpoints:
    """Tests for GET, POST, DELETE /employees."""

    def test_list_employees(self, client: TestClient):
        response = client.get("/employees")
        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert len(data["employees"]) == 1
        emp = data["employees"][0]
        assert emp["employee_id"] == "EMP001"
        assert emp["employee_name"] == "Jane Doe"
        assert emp["sample_count"] == 3
        assert emp["embedding_dim"] == 128

    def test_get_existing_employee(self, client: TestClient):
        response = client.get("/employees/EMP001")
        assert response.status_code == 200
        emp = response.json()
        assert emp["employee_id"] == "EMP001"
        assert emp["employee_name"] == "Jane Doe"

    def test_get_nonexistent_employee_404(self, client: TestClient):
        response = client.get("/employees/NON_EXISTENT")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_register_employee_success(self, client: TestClient):
        img_bytes = make_test_image_bytes()
        files = [
            ("images", ("face1.png", io.BytesIO(img_bytes), "image/png")),
            ("images", ("face2.png", io.BytesIO(img_bytes), "image/png")),
        ]
        data = {
            "employee_id": "EMP002",
            "employee_name": "John Doe",
        }

        response = client.post("/employees", data=data, files=files)
        assert response.status_code == 201
        res = response.json()
        assert res["success"] is True
        assert res["employee_id"] == "EMP002"
        assert res["employee_name"] == "John Doe"
        assert res["valid_samples"] == 2
        assert res["rejected_samples"] == 0

    def test_register_employee_empty_id_400(self, client: TestClient):
        img_bytes = make_test_image_bytes()
        files = [("images", ("face1.png", io.BytesIO(img_bytes), "image/png"))]
        data = {"employee_id": "   ", "employee_name": "Valid Name"}

        response = client.post("/employees", data=data, files=files)
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_register_employee_empty_name_400(self, client: TestClient):
        img_bytes = make_test_image_bytes()
        files = [("images", ("face1.png", io.BytesIO(img_bytes), "image/png"))]
        data = {"employee_id": "EMP003", "employee_name": ""}

        response = client.post("/employees", data=data, files=files)
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_delete_existing_employee(self, client: TestClient):
        response = client.delete("/employees/EMP001")
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["detail"]

    def test_delete_nonexistent_employee_404(self, client: TestClient):
        response = client.delete("/employees/DOES_NOT_EXIST")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Monitoring Tests
# ---------------------------------------------------------------------------


class TestMonitoringEndpoints:
    """Tests for GET /monitoring/status, POST /monitoring/start, POST /monitoring/stop."""

    def test_get_monitoring_status_structured_format(self, client: TestClient):
        response = client.get("/monitoring/status")
        assert response.status_code == 200
        data = response.json()

        assert "is_active" in data
        assert "total_tracked" in data
        assert "compliant_count" in data
        assert "non_compliant_count" in data
        assert "persons" in data
        assert "telemetry" in data

        assert len(data["persons"]) == 1
        person = data["persons"][0]

        # Verify all required structured keys
        required_keys = {
            "track_id",
            "employee_id",
            "employee_name",
            "hardhat",
            "safety_vest",
            "compliance_status",
            "missing_items",
            "detection_confidence",
        }
        assert required_keys.issubset(person.keys())

        assert person["track_id"] == "track_1"
        assert person["employee_id"] == "EMP001"
        assert person["employee_name"] == "Jane Doe"
        assert person["hardhat"] is True
        assert person["safety_vest"] is True
        assert person["compliance_status"] == "COMPLIANT"
        assert person["missing_items"] == []
        assert person["detection_confidence"] == 0.92

    def test_monitoring_start(self, client: TestClient):
        response = client.post("/monitoring/start", json={"source": 0})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["source"] == "0"

    def test_monitoring_stop(self, client: TestClient):
        response = client.post("/monitoring/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"

    def test_toggle_camera_flip(self, client: TestClient):
        response = client.post("/monitoring/flip", json={"flip_horizontal": True, "flip_vertical": False})
        assert response.status_code == 200
        data = response.json()
        assert data["flip_horizontal"] is True
        assert data["flip_vertical"] is False

    def test_get_camera_snapshot_success(self, client: TestClient, mock_monitoring_service: MagicMock):
        mock_monitoring_service.get_latest_raw_frame_jpeg.return_value = b"\xff\xd8\xff\xe0fake_jpeg"
        response = client.get("/monitoring/snapshot?raw=true")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"\xff\xd8\xff\xe0fake_jpeg"

    def test_get_camera_snapshot_not_found(self, client: TestClient, mock_monitoring_service: MagicMock):
        mock_monitoring_service.get_latest_raw_frame_jpeg.return_value = None
        response = client.get("/monitoring/snapshot?raw=true")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket Tests
# ---------------------------------------------------------------------------


class TestWebSocketMonitoring:
    """Tests for WebSocket /ws/monitoring and /monitoring/ws."""

    def test_websocket_receives_structured_snapshot(self, client: TestClient):
        """Verifies WebSocket accepts connection, sends initial structured snapshot, and does not send raw frames."""
        with client.websocket_connect("/ws/monitoring") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            data = msg["data"]

            assert "persons" in data
            assert len(data["persons"]) == 1
            assert data["persons"][0]["track_id"] == "track_1"
            assert data["persons"][0]["employee_id"] == "EMP001"

            # Verify no raw video or base64 frames are included
            assert "frame" not in data
            assert "raw_frame" not in data
            assert "image" not in data

            # Heartbeat ping
            ws.send_text("ping")
            pong = ws.receive_json()
            assert pong == {"type": "pong"}

    def test_websocket_alias_endpoint(self, client: TestClient):
        """Verifies /monitoring/ws alias functions identically."""
        with client.websocket_connect("/monitoring/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            assert msg["data"]["total_tracked"] == 1


# ---------------------------------------------------------------------------
# CORS Tests
# ---------------------------------------------------------------------------


class TestCORSConfiguration:
    """Verifies CORS headers for React frontend integration."""

    def test_cors_preflight(self, client: TestClient):
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = client.options("/employees", headers=headers)
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] in ("*", "http://localhost:5173")
