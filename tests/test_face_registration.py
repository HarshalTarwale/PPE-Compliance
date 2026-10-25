"""Unit tests for employee face registration and recognition (recognition/registration.py).

Tests:
  1. Face embedding extraction and cosine similarity calculations.
  2. Single face validation (accepts 1 face, rejects 0 faces and multiple faces).
  3. Rejection of ambiguous registration images.
  4. Local employee storage persistence (record.json and embeddings.npz, NO raw images).
  5. Multi-image enrollment and normalized mean embedding calculation.
  6. Threshold-based identification (matches known, strictly rejects unknown persons).
  7. Configurable recognition thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from recognition.embeddings import (
    DEFAULT_COSINE_THRESHOLD,
    FaceDetection,
    FaceEmbeddingEngine,
)
from recognition.registration import (
    EmployeeRecord,
    EmployeeRegistrationService,
    EmployeeStorage,
    RegistrationResult,
    RegistrationStatus,
)


# ---------------------------------------------------------------------------
# Embedding Engine Tests
# ---------------------------------------------------------------------------


class TestFaceEmbeddingEngine:
    """Test embedding engine utilities."""

    @pytest.fixture
    def engine(self) -> FaceEmbeddingEngine:
        return FaceEmbeddingEngine()

    def test_engine_initializes(self, engine: FaceEmbeddingEngine):
        assert engine.detector is not None
        assert engine.recognizer is not None
        assert engine.cosine_threshold == DEFAULT_COSINE_THRESHOLD

    def test_compute_similarity_identical_vectors(self, engine: FaceEmbeddingEngine):
        v = np.random.randn(128).astype(np.float32)
        v = v / np.linalg.norm(v)
        sim = engine.compute_similarity(v, v)
        assert pytest.approx(sim, 0.001) == 1.0

    def test_compute_similarity_orthogonal_vectors(self, engine: FaceEmbeddingEngine):
        v1 = np.zeros(128, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(128, dtype=np.float32)
        v2[1] = 1.0
        sim = engine.compute_similarity(v1, v2)
        assert pytest.approx(sim, 0.001) == 0.0

    def test_is_match_threshold(self, engine: FaceEmbeddingEngine):
        v = np.random.randn(128).astype(np.float32)
        v = v / np.linalg.norm(v)

        # Identical vector matches with default threshold
        assert engine.is_match(v, v, threshold=0.50) is True

        # Orthogonal vector does not match
        v_ortho = np.zeros(128, dtype=np.float32)
        v_ortho[0] = 1.0
        v_diff = np.zeros(128, dtype=np.float32)
        v_diff[1] = 1.0
        assert engine.is_match(v_ortho, v_diff, threshold=0.363) is False

    def test_concurrent_multithreaded_detection(self, engine: FaceEmbeddingEngine):
        """Verify thread-safety when detect_faces is called concurrently with different image shapes."""
        import threading

        errors = []

        def worker(shape):
            try:
                for _ in range(15):
                    dummy = np.zeros(shape, dtype=np.uint8)
                    engine.detect_faces(dummy)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=((480, 640, 3),))
        t2 = threading.Thread(target=worker, args=((720, 1280, 3),))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Concurrent face detection failed with errors: {errors}"


# ---------------------------------------------------------------------------
# Storage Tests (Verifying NO Raw Images Stored)
# ---------------------------------------------------------------------------


class TestEmployeeStorage:
    """Test local storage and privacy-preserving persistence."""

    @pytest.fixture
    def temp_storage(self, tmp_path: Path) -> EmployeeStorage:
        return EmployeeStorage(storage_dir=tmp_path / "employees")

    def test_save_and_load_record(self, temp_storage: EmployeeStorage):
        emb1 = np.random.randn(128).astype(np.float32)
        emb2 = np.random.randn(128).astype(np.float32)
        mean_emb = (emb1 + emb2) / 2.0
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        record = EmployeeRecord(
            employee_id="EMP001",
            employee_name="Alice Worker",
            registered_at="2026-09-25T12:00:00",
            sample_count=2,
            mean_embedding=mean_emb,
            sample_embeddings=[emb1, emb2],
            metadata={"department": "Safety"},
        )

        emp_dir = temp_storage.save_record(record)
        assert emp_dir.is_dir()

        # Verify only record.json and embeddings.npz exist (NO raw images)
        files = list(emp_dir.iterdir())
        file_names = {f.name for f in files}
        assert file_names == {"record.json", "embeddings.npz"}

        # Load and verify
        loaded = temp_storage.load_record("EMP001")
        assert loaded is not None
        assert loaded.employee_id == "EMP001"
        assert loaded.employee_name == "Alice Worker"
        assert loaded.sample_count == 2
        assert loaded.mean_embedding.shape == (128,)
        assert len(loaded.sample_embeddings) == 2
        assert loaded.metadata["department"] == "Safety"

    def test_list_records_and_delete(self, temp_storage: EmployeeStorage):
        v = np.zeros(128, dtype=np.float32)
        rec = EmployeeRecord("EMP002", "Bob", "now", 1, v, [v])
        temp_storage.save_record(rec)

        assert temp_storage.exists("EMP002") is True
        assert len(temp_storage.list_records()) == 1

        deleted = temp_storage.delete_record("EMP002")
        assert deleted is True
        assert temp_storage.exists("EMP002") is False
        assert len(temp_storage.list_records()) == 0


# ---------------------------------------------------------------------------
# Registration Service & Ambiguity Handling Tests
# ---------------------------------------------------------------------------


class TestEmployeeRegistrationService:
    """Test enrollment validation, ambiguity rejection, and identification."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> EmployeeRegistrationService:
        storage = EmployeeStorage(storage_dir=tmp_path / "employees")
        mock_engine = MagicMock(spec=FaceEmbeddingEngine)
        mock_engine.cosine_threshold = 0.363
        return EmployeeRegistrationService(storage=storage, embedding_engine=mock_engine)

    def test_empty_id_or_name_rejected(self, service: EmployeeRegistrationService):
        res1 = service.register_employee("", "Valid Name", [np.zeros((100, 100, 3), dtype=np.uint8)])
        assert res1.success is False
        assert "ID cannot be empty" in res1.message

        res2 = service.register_employee("EMP01", "   ", [np.zeros((100, 100, 3), dtype=np.uint8)])
        assert res2.success is False
        assert "name cannot be empty" in res2.message

    def test_reject_ambiguous_images(self, service: EmployeeRegistrationService):
        """Image 1: 0 faces (rejected)
        Image 2: 2 faces (rejected - multiple)
        Image 3: 1 face (accepted)"""
        valid_emb = np.random.randn(128).astype(np.float32)
        valid_emb = valid_emb / np.linalg.norm(valid_emb)

        service.engine.extract_single_face_embedding.side_effect = [
            (None, "NO_FACE_DETECTED", 0),
            (None, "MULTIPLE_FACES_DETECTED", 2),
            (valid_emb, "SUCCESS", 1),
        ]

        dummy_imgs = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
        ]

        result = service.register_employee("EMP003", "Charlie", dummy_imgs)

        assert result.success is True
        assert result.valid_samples == 1
        assert result.rejected_samples == 2
        assert result.evaluations[0].status == RegistrationStatus.NO_FACE_DETECTED
        assert result.evaluations[1].status == RegistrationStatus.MULTIPLE_FACES_DETECTED
        assert result.evaluations[2].status == RegistrationStatus.SUCCESS

    def test_all_images_rejected_fails_registration(self, service: EmployeeRegistrationService):
        service.engine.extract_single_face_embedding.return_value = (None, "NO_FACE_DETECTED", 0)
        dummy_imgs = [np.zeros((100, 100, 3), dtype=np.uint8)]

        result = service.register_employee("EMP004", "David", dummy_imgs)
        assert result.success is False
        assert result.valid_samples == 0
        assert "Registration failed" in result.message
        assert service.storage.exists("EMP004") is False

    def test_identify_known_person(self, service: EmployeeRegistrationService):
        # Register an employee
        target_emb = np.random.randn(128).astype(np.float32)
        target_emb = target_emb / np.linalg.norm(target_emb)

        service.engine.extract_single_face_embedding.return_value = (target_emb, "SUCCESS", 1)
        service.engine.compute_similarity.return_value = 0.88  # High similarity

        service.register_employee("EMP005", "Emma Stone", [np.zeros((100, 100, 3), dtype=np.uint8)])

        # Identify with target embedding
        match = service.identify(target_emb, threshold=0.363)

        assert match is not None
        assert match["employee_id"] == "EMP005"
        assert match["employee_name"] == "Emma Stone"
        assert match["similarity"] == 0.88

    def test_identify_unknown_person_strictly_returns_none(self, service: EmployeeRegistrationService):
        """Unknown person (similarity below threshold) must NOT be guessed as registered."""
        target_emb = np.random.randn(128).astype(np.float32)
        target_emb = target_emb / np.linalg.norm(target_emb)

        service.engine.extract_single_face_embedding.return_value = (target_emb, "SUCCESS", 1)
        service.register_employee("EMP006", "Frank", [np.zeros((100, 100, 3), dtype=np.uint8)])

        # Query embedding yields very low similarity (unknown stranger)
        service.engine.compute_similarity.return_value = 0.12  # Below 0.363 threshold

        stranger_emb = np.random.randn(128).astype(np.float32)
        match = service.identify(stranger_emb, threshold=0.363)

        # Must strictly be None (do not automatically identify unknown people)
        assert match is None

    def test_identify_rejects_ambiguous_margin(self, service: EmployeeRegistrationService):
        """When two employees score closely (margin < 0.09), reject match as ambiguous."""
        emb_a = np.random.randn(128).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)
        emb_b = np.random.randn(128).astype(np.float32)
        emb_b = emb_b / np.linalg.norm(emb_b)

        service.engine.extract_single_face_embedding.side_effect = [
            (emb_a, "SUCCESS", 1),
            (emb_b, "SUCCESS", 1),
        ]
        service.register_employee("EMP_A", "Alice", [np.zeros((100, 100, 3), dtype=np.uint8)])
        service.register_employee("EMP_B", "Bob", [np.zeros((100, 100, 3), dtype=np.uint8)])

        # Query scores 0.62 for Alice and 0.58 for Bob (margin = 0.04 < MATCH_MARGIN 0.09)
        def mock_similarity(q, target):
            if np.allclose(target, emb_a, atol=1e-5):
                return 0.62
            elif np.allclose(target, emb_b, atol=1e-5):
                return 0.58
            return 0.0

        service.engine.compute_similarity.side_effect = mock_similarity

        query_emb = np.random.randn(128).astype(np.float32)
        match = service.identify(query_emb, threshold=0.50)
        assert match is None, "Ambiguous match with margin < 0.09 must be rejected"

    def test_identify_accepts_when_margin_is_sufficient(self, service: EmployeeRegistrationService):
        """When the best match clearly outscores the runner-up (margin >= 0.09), accept."""
        emb_a = np.random.randn(128).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)
        emb_b = np.random.randn(128).astype(np.float32)
        emb_b = emb_b / np.linalg.norm(emb_b)

        service.engine.extract_single_face_embedding.side_effect = [
            (emb_a, "SUCCESS", 1),
            (emb_b, "SUCCESS", 1),
        ]
        service.register_employee("EMP_A", "Alice", [np.zeros((100, 100, 3), dtype=np.uint8)])
        service.register_employee("EMP_B", "Bob", [np.zeros((100, 100, 3), dtype=np.uint8)])

        # Alice: 0.72, Bob: 0.55 (margin = 0.17 >= 0.09)
        def mock_similarity(q, target):
            if np.allclose(target, emb_a, atol=1e-5):
                return 0.72
            elif np.allclose(target, emb_b, atol=1e-5):
                return 0.55
            return 0.0

        service.engine.compute_similarity.side_effect = mock_similarity

        query_emb = np.random.randn(128).astype(np.float32)
        match = service.identify(query_emb, threshold=0.50)
        assert match is not None
        assert match["employee_id"] == "EMP_A"
        assert match["employee_name"] == "Alice"
        assert match["similarity"] == 0.72

    def test_identify_uses_mean_embedding_only(self, service: EmployeeRegistrationService):
        """Verification must only compare against mean embedding, preventing single-sample score inflation."""
        emb1 = np.random.randn(128).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = np.random.randn(128).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        service.engine.extract_single_face_embedding.side_effect = [
            (emb1, "SUCCESS", 1),
            (emb2, "SUCCESS", 1),
        ]
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        service.register_employee("EMP_MULTI", "MultiSample", [dummy, dummy])

        rec = service.storage.load_record("EMP_MULTI")
        assert rec is not None
        assert len(rec.sample_embeddings) == 2

        service.engine.compute_similarity.reset_mock()
        service.engine.compute_similarity.return_value = 0.85

        query_emb = np.random.randn(128).astype(np.float32)
        match = service.identify(query_emb, threshold=0.50)

        # Ensure compute_similarity was called exactly ONCE per employee (mean only)
        # and NOT called for individual samples
        assert service.engine.compute_similarity.call_count == 1
        called_target = service.engine.compute_similarity.call_args[0][1]
        assert np.array_equal(called_target, rec.mean_embedding)
        assert match is not None
        assert match["employee_id"] == "EMP_MULTI"
