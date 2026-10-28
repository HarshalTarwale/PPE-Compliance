"""
Tests for safety event history, unknown-person alerts, and identity revocation.

Covers:
  1. SafetyEventLog persistence (append, list, filter, clear, reload)
  2. MonitoringService event detection (violation confirm/dedupe/reset, unknown alerts)
  3. GET / DELETE /monitoring/events routes
  4. Pipeline drops the identity of an employee deleted from the database
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.schemas.monitoring import MonitoringStatusResponse, TrackedPersonSchema
from backend.services.event_log import SafetyEventLog
from backend.services.monitoring_service import MonitoringService
from detection.detector import Detection, DetectionResult, PPEDetector
from detection.pipeline import PPECompliancePipeline
from recognition.registration import EmployeeRegistrationService


def make_person(
    track_id: str = "track_1",
    employee_id: str = "UNKNOWN",
    employee_name: str = "UNKNOWN",
    missing: List[str] = (),
) -> TrackedPersonSchema:
    missing = list(missing)
    return TrackedPersonSchema(
        track_id=track_id,
        employee_id=employee_id,
        employee_name=employee_name,
        hardhat="Hardhat" not in missing,
        safety_vest="Safety Vest" not in missing,
        compliance_status="NON_COMPLIANT" if missing else "COMPLIANT",
        missing_items=missing,
        detection_confidence=0.9,
        bbox=[0, 0, 100, 200],
        is_compliant=not missing,
    )


def make_status(persons: List[TrackedPersonSchema]) -> MonitoringStatusResponse:
    return MonitoringStatusResponse(
        is_active=True,
        source="browser_camera",
        total_tracked=len(persons),
        compliant_count=sum(p.is_compliant for p in persons),
        non_compliant_count=sum(not p.is_compliant for p in persons),
        persons=persons,
        timestamp="2026-10-28T10:00:00Z",
    )


@pytest.fixture
def event_log(tmp_path) -> SafetyEventLog:
    return SafetyEventLog(tmp_path / "events.jsonl")


@pytest.fixture
def service(event_log: SafetyEventLog) -> MonitoringService:
    return MonitoringService(pipeline=MagicMock(), event_log=event_log)


# ---------------------------------------------------------------------------
# SafetyEventLog
# ---------------------------------------------------------------------------


class TestSafetyEventLog:
    def test_append_list_newest_first_and_reload(self, tmp_path):
        path = tmp_path / "log.jsonl"
        log = SafetyEventLog(path)
        log.append({"id": "a", "type": "ppe_violation"})
        log.append({"id": "b", "type": "unknown_person"})

        assert [e["id"] for e in log.list()] == ["b", "a"]
        assert [e["id"] for e in log.list(event_type="ppe_violation")] == ["a"]

        reloaded = SafetyEventLog(path)
        assert reloaded.count() == 2

    def test_clear_removes_everything(self, event_log: SafetyEventLog):
        event_log.append({"id": "a", "type": "ppe_violation"})
        assert event_log.clear() == 1
        assert event_log.list() == []
        assert SafetyEventLog(event_log.path).count() == 0


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


class TestEventDetection:
    def test_violation_logged_after_confirmation_and_only_once(self, service: MonitoringService):
        status = make_status([make_person(employee_id="EMP4", employee_name="Harshal", missing=["Hardhat"])])

        assert service._detect_events(status, now=0.0) == []  # not yet confirmed
        events = service._detect_events(status, now=1.5)
        assert len(events) == 1
        assert events[0].type == "ppe_violation"
        assert events[0].employee_name == "Harshal"
        assert events[0].missing_items == ["Hardhat"]
        assert service._detect_events(status, now=3.0) == []  # same violation, not repeated

        stored = service.list_events()
        assert stored["total"] == 1
        assert stored["events"][0]["message"] == "Harshal is missing: Hardhat"

    def test_new_missing_item_logs_again(self, service: MonitoringService):
        p1 = make_person(employee_id="EMP4", employee_name="Harshal", missing=["Hardhat"])
        p2 = make_person(employee_id="EMP4", employee_name="Harshal", missing=["Hardhat", "Safety Vest"])
        service._detect_events(make_status([p1]), now=0.0)
        assert len(service._detect_events(make_status([p1]), now=1.5)) == 1
        assert len(service._detect_events(make_status([p2]), now=2.0)) == 1

    def test_violation_relogged_only_after_sustained_compliance(self, service: MonitoringService):
        bad = make_status([make_person(employee_id="EMP4", employee_name="H", missing=["Hardhat"])])
        good = make_status([make_person(employee_id="EMP4", employee_name="H")])

        service._detect_events(bad, now=0.0)
        assert len(service._detect_events(bad, now=1.0)) == 1
        # Brief compliance flicker: same violation is not logged again
        service._detect_events(good, now=2.0)
        service._detect_events(bad, now=3.0)
        assert service._detect_events(bad, now=4.5) == []
        # Compliant for longer than the reset window: a new violation is logged
        service._detect_events(good, now=5.0)
        service._detect_events(good, now=11.0)
        service._detect_events(bad, now=12.0)
        assert len(service._detect_events(bad, now=13.0)) == 1

    def test_unknown_person_alert_once_per_track(self, service: MonitoringService):
        status = make_status([make_person(track_id="track_7")])

        assert service._detect_events(status, now=0.0) == []
        assert service._detect_events(status, now=1.0) == []  # under 2 seconds
        events = service._detect_events(status, now=2.5)
        assert [e.type for e in events] == ["unknown_person"]
        assert events[0].track_id == "track_7"
        assert service._detect_events(status, now=10.0) == []

    def test_recognized_person_raises_no_unknown_alert(self, service: MonitoringService):
        status = make_status([make_person(employee_id="EMP1", employee_name="Bharat")])
        for t in (0.0, 1.0, 3.0, 6.0):
            assert service._detect_events(status, now=t) == []

    def test_departed_track_state_is_pruned(self, service: MonitoringService):
        service._detect_events(make_status([make_person(track_id="track_1")]), now=0.0)
        service._detect_events(make_status([]), now=1.0)
        assert service._alert_states == {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestEventRoutes:
    def test_list_and_clear_events(self, service: MonitoringService):
        app = create_app(employee_service=MagicMock(), monitoring_service=service)
        status = make_status([make_person(track_id="track_3", missing=["Safety Vest"])])
        service._detect_events(status, now=0.0)
        service._detect_events(status, now=2.5)

        with TestClient(app) as client:
            res = client.get("/monitoring/events")
            assert res.status_code == 200
            body = res.json()
            assert body["total"] == 2
            assert {e["type"] for e in body["events"]} == {"ppe_violation", "unknown_person"}

            res = client.get("/monitoring/events", params={"type": "unknown_person"})
            assert [e["type"] for e in res.json()["events"]] == ["unknown_person"]

            res = client.delete("/monitoring/events")
            assert res.status_code == 200
            assert res.json()["removed"] == 2
            assert client.get("/monitoring/events").json()["total"] == 0


# ---------------------------------------------------------------------------
# Identity revocation for deleted employees
# ---------------------------------------------------------------------------


class TestDeletedEmployeeRevocation:
    def test_deleted_employee_becomes_unknown_immediately(self):
        det = Detection(x1=50, y1=50, x2=200, y2=400, class_id=0, class_name="Person", confidence=0.95)
        detector = MagicMock(spec=PPEDetector)
        detector.detect.return_value = DetectionResult(detections=[det])

        face_service = MagicMock(spec=EmployeeRegistrationService)
        face_service.identify.return_value = {
            "employee_id": "EMP4", "employee_name": "Harshal", "similarity": 0.8,
        }
        registered = {"EMP4"}
        face_service.registered_employee_ids.side_effect = lambda: set(registered)

        pipeline = PPECompliancePipeline(detector=detector, face_service=face_service)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        assert pipeline.process_frame(frame).records[0].employee_name == "Harshal"

        # Employee deleted; the face no longer matches anyone
        registered.clear()
        face_service.identify.return_value = None
        rec = pipeline.process_frame(frame).records[0]
        assert rec.employee_id == "UNKNOWN"
        assert rec.employee_name == "UNKNOWN"
        assert rec.face_similarity is None
