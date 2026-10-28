"""
Employee Face Registration and Database Management Service.

Provides a robust employee face enrollment and identity management pipeline:
  1. Ingests multiple face images per employee.
  2. Detects and aligns faces (YuNet + SFace).
  3. Rejects ambiguous images (images with 0 faces or >1 faces).
  4. Extracts 128-D normalized embeddings.
  5. Computes an aggregate normalized mean representation.
  6. Stores embeddings and metadata locally WITHOUT retaining raw images.
  7. Identifies registered employees while strictly rejecting unknown persons.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

# Ensure project root is importable
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from recognition.embeddings import (
    DEFAULT_COSINE_THRESHOLD,
    FaceEmbeddingEngine,
)

logger = logging.getLogger("registration")

# Default local storage location for face database
DEFAULT_STORAGE_DIR = _PROJECT_ROOT / "data" / "employees"

# Default cosine similarity threshold for open-set identification.
# 0.50 provides high discrimination against false acceptances in surveillance environments.
DEFAULT_IDENTIFICATION_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Enums and Data Containers
# ---------------------------------------------------------------------------


class RegistrationStatus(str, Enum):
    """Status codes for image processing during registration."""

    SUCCESS = "SUCCESS"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    UNREADABLE_IMAGE = "UNREADABLE_IMAGE"


@dataclass
class SampleEvaluation:
    """Report for an individual image submitted during registration."""

    sample_index: int
    source_label: str
    status: RegistrationStatus
    face_count: int
    message: str


@dataclass
class EmployeeRecord:
    """Stored representation of an enrolled employee."""

    employee_id: str
    employee_name: str
    registered_at: str
    sample_count: int
    mean_embedding: np.ndarray  # (128,) float32 unit-normalized vector
    sample_embeddings: List[np.ndarray] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary (omitting raw embedding arrays)."""
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "registered_at": self.registered_at,
            "sample_count": self.sample_count,
            "embedding_dim": int(self.mean_embedding.shape[0]),
            "metadata": self.metadata,
        }


@dataclass
class RegistrationResult:
    """Result of an employee registration attempt."""

    success: bool
    employee_id: str
    employee_name: str
    valid_samples: int
    rejected_samples: int
    evaluations: List[SampleEvaluation] = field(default_factory=list)
    message: str = ""
    record: Optional[EmployeeRecord] = None


# ---------------------------------------------------------------------------
# Local Storage Engine
# ---------------------------------------------------------------------------


class EmployeeStorage:
    """Manages local persistence of employee embeddings and metadata.

    Directory structure:
      <storage_dir>/
        catalog.json               <- List of all enrolled employee metadata
        <employee_id>/
          record.json              <- Employee metadata details
          embeddings.npz           <- mean_embedding and sample_embeddings arrays

    Note: Raw face images are NEVER stored on disk to protect privacy.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize employee storage.

        Args:
            storage_dir: Base directory for storing employee records.
        """
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.storage_dir / "catalog.json"
        self._cached_records: Optional[List[EmployeeRecord]] = None
        self._cached_mtime: float = 0.0
        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        """Create empty catalog.json if it doesn't already exist."""
        if not self.catalog_path.is_file():
            self._save_catalog({})

    def _load_catalog(self) -> Dict[str, Any]:
        """Load catalog dictionary."""
        if not self.catalog_path.is_file():
            return {}
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.error("Failed to read employee catalog: %s", err)
            return {}

    def _save_catalog(self, catalog: Dict[str, Any]) -> None:
        """Save catalog dictionary."""
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    def exists(self, employee_id: str) -> bool:
        """Check if an employee ID is already registered."""
        catalog = self._load_catalog()
        return employee_id in catalog

    def save_record(self, record: EmployeeRecord) -> Path:
        """Save employee record and compressed embeddings to disk.

        Args:
            record: EmployeeRecord instance.

        Returns:
            Path to the employee record folder.
        """
        emp_dir = self.storage_dir / record.employee_id
        emp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata JSON
        record_json_path = emp_dir / "record.json"
        with open(record_json_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        # 2. Save embeddings in compressed .npz archive
        npz_path = emp_dir / "embeddings.npz"
        np.savez_compressed(
            npz_path,
            mean_embedding=record.mean_embedding,
            sample_embeddings=np.array(record.sample_embeddings),
        )

        # 3. Update global catalog
        catalog = self._load_catalog()
        catalog[record.employee_id] = record.to_dict()
        self._save_catalog(catalog)
        self._cached_records = None

        logger.info(
            "Saved employee record for '%s' (%s) with %d samples.",
            record.employee_name,
            record.employee_id,
            record.sample_count,
        )
        return emp_dir

    def load_record(self, employee_id: str) -> Optional[EmployeeRecord]:
        """Load an employee record and embeddings by ID.

        Args:
            employee_id: Identifier of the employee.

        Returns:
            EmployeeRecord if found, None otherwise.
        """
        emp_dir = self.storage_dir / employee_id
        record_json_path = emp_dir / "record.json"
        npz_path = emp_dir / "embeddings.npz"

        if not (record_json_path.is_file() and npz_path.is_file()):
            return None

        try:
            with open(record_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with np.load(npz_path) as npz_data:
                mean_emb = npz_data["mean_embedding"]
                sample_embs_arr = npz_data["sample_embeddings"]
                sample_embs = [sample_embs_arr[i] for i in range(len(sample_embs_arr))]

            return EmployeeRecord(
                employee_id=data["employee_id"],
                employee_name=data["employee_name"],
                registered_at=data["registered_at"],
                sample_count=data["sample_count"],
                mean_embedding=mean_emb,
                sample_embeddings=sample_embs,
                metadata=data.get("metadata", {}),
            )
        except Exception as err:
            logger.error("Error loading record for %s: %s", employee_id, err)
            return None

    def list_records(self) -> List[Dict[str, Any]]:
        """Return list of all registered employee metadata from catalog."""
        catalog = self._load_catalog()
        return list(catalog.values())

    def load_all_records(self) -> List[EmployeeRecord]:
        """Load all full employee records into memory for fast matching."""
        current_mtime = self.catalog_path.stat().st_mtime if self.catalog_path.is_file() else 0.0
        if (
            getattr(self, "_cached_records", None) is not None
            and getattr(self, "_cached_mtime", None) == current_mtime
        ):
            return self._cached_records

        catalog = self._load_catalog()
        records: List[EmployeeRecord] = []
        for emp_id in catalog.keys():
            rec = self.load_record(emp_id)
            if rec is not None:
                records.append(rec)
        self._cached_records = records
        self._cached_mtime = current_mtime
        return records

    def delete_record(self, employee_id: str) -> bool:
        """Delete an employee's record and embeddings from storage.

        Args:
            employee_id: ID of employee to remove.

        Returns:
            True if record was deleted, False if not found.
        """
        catalog = self._load_catalog()
        if employee_id not in catalog:
            return False

        emp_dir = self.storage_dir / employee_id
        if emp_dir.is_dir():
            for f in emp_dir.iterdir():
                f.unlink(missing_ok=True)
            emp_dir.rmdir()

        del catalog[employee_id]
        self._save_catalog(catalog)
        self._cached_records = None
        logger.info("Deleted employee record: %s", employee_id)
        return True


# ---------------------------------------------------------------------------
# Employee Registration Service
# ---------------------------------------------------------------------------


class EmployeeRegistrationService:
    """Service orchestrating employee enrollment, validation, and identity matching."""

    def __init__(
        self,
        storage: Optional[EmployeeStorage] = None,
        embedding_engine: Optional[FaceEmbeddingEngine] = None,
        default_threshold: float = DEFAULT_IDENTIFICATION_THRESHOLD,
    ) -> None:
        """Initialize registration service.

        Args:
            storage: Storage backend for employee records.
            embedding_engine: Face detection and embedding engine.
            default_threshold: Cosine similarity threshold for positive recognition.
        """
        self.storage = storage or EmployeeStorage()
        self.engine = embedding_engine or FaceEmbeddingEngine()
        self.default_threshold = float(default_threshold)

    # -----------------------------------------------------------------------
    # Registration / Enrollment
    # -----------------------------------------------------------------------

    def register_employee(
        self,
        employee_id: str,
        employee_name: str,
        images: List[Union[str, Path, np.ndarray]],
        allow_overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegistrationResult:
        """Enroll an employee from multiple face images.

        Pipeline:
          1. Validates non-empty ID and Name.
          2. Processes each image:
             - Rejects if 0 faces detected (NO_FACE_DETECTED)
             - Rejects if >1 faces detected (MULTIPLE_FACES_DETECTED - ambiguous)
             - Extracts aligned 128-D embedding if exactly 1 face is found.
          3. Computes normalized mean embedding across all valid samples.
          4. Persists record and embeddings without saving raw images.

        Args:
            employee_id: Unique identifier for employee (e.g. 'EMP001').
            employee_name: Full name of employee.
            images: List of image file paths or BGR numpy arrays.
            allow_overwrite: If True, updates existing record.
            metadata: Optional additional employee metadata.

        Returns:
            RegistrationResult containing status and sample breakdown.
        """
        clean_id = employee_id.strip()
        clean_name = employee_name.strip()

        if not clean_id:
            return RegistrationResult(
                success=False,
                employee_id="",
                employee_name=clean_name,
                valid_samples=0,
                rejected_samples=0,
                message="Employee ID cannot be empty.",
            )

        if not clean_name:
            return RegistrationResult(
                success=False,
                employee_id=clean_id,
                employee_name="",
                valid_samples=0,
                rejected_samples=0,
                message="Employee name cannot be empty.",
            )

        if not allow_overwrite and self.storage.exists(clean_id):
            return RegistrationResult(
                success=False,
                employee_id=clean_id,
                employee_name=clean_name,
                valid_samples=0,
                rejected_samples=0,
                message=f"Employee ID '{clean_id}' is already registered. Set allow_overwrite=True to update.",
            )

        if not images:
            return RegistrationResult(
                success=False,
                employee_id=clean_id,
                employee_name=clean_name,
                valid_samples=0,
                rejected_samples=0,
                message="At least one enrollment face image must be provided.",
            )

        valid_embeddings: List[np.ndarray] = []
        evaluations: List[SampleEvaluation] = []

        # Process each image
        for idx, img_input in enumerate(images):
            src_label = str(img_input) if isinstance(img_input, (str, Path)) else f"frame_{idx + 1}"

            # Load image if path
            if isinstance(img_input, (str, Path)):
                img_path = Path(img_input)
                if not img_path.is_file():
                    evaluations.append(
                        SampleEvaluation(
                            sample_index=idx,
                            source_label=src_label,
                            status=RegistrationStatus.UNREADABLE_IMAGE,
                            face_count=0,
                            message=f"File not found: {img_path}",
                        )
                    )
                    continue

                bgr_img = cv2.imread(str(img_path))
                if bgr_img is None:
                    evaluations.append(
                        SampleEvaluation(
                            sample_index=idx,
                            source_label=src_label,
                            status=RegistrationStatus.UNREADABLE_IMAGE,
                            face_count=0,
                            message="Failed to decode image file.",
                        )
                    )
                    continue
            elif isinstance(img_input, np.ndarray):
                bgr_img = img_input
            else:
                evaluations.append(
                    SampleEvaluation(
                        sample_index=idx,
                        source_label=src_label,
                        status=RegistrationStatus.UNREADABLE_IMAGE,
                        face_count=0,
                        message=f"Unsupported image type: {type(img_input)}",
                    )
                )
                continue

            # Run detection & validation
            embedding, status_str, face_count = self.engine.extract_single_face_embedding(bgr_img)

            if status_str == "NO_FACE_DETECTED":
                evaluations.append(
                    SampleEvaluation(
                        sample_index=idx,
                        source_label=src_label,
                        status=RegistrationStatus.NO_FACE_DETECTED,
                        face_count=0,
                        message="Rejected: No face was detected in this image.",
                    )
                )
            elif status_str == "MULTIPLE_FACES_DETECTED":
                evaluations.append(
                    SampleEvaluation(
                        sample_index=idx,
                        source_label=src_label,
                        status=RegistrationStatus.MULTIPLE_FACES_DETECTED,
                        face_count=face_count,
                        message=f"Rejected: Image contains {face_count} faces. Registration requires exactly one face.",
                    )
                )
            elif status_str == "SUCCESS" and embedding is not None:
                valid_embeddings.append(embedding)
                evaluations.append(
                    SampleEvaluation(
                        sample_index=idx,
                        source_label=src_label,
                        status=RegistrationStatus.SUCCESS,
                        face_count=1,
                        message="Accepted: Single face detected and aligned.",
                    )
                )

        valid_count = len(valid_embeddings)
        rejected_count = len(evaluations) - valid_count

        # If zero valid samples, registration fails
        if valid_count == 0:
            reasons = [f"Image #{e.sample_index + 1} ({e.status.value}): {e.message}" for e in evaluations]
            return RegistrationResult(
                success=False,
                employee_id=clean_id,
                employee_name=clean_name,
                valid_samples=0,
                rejected_samples=rejected_count,
                evaluations=evaluations,
                message=f"Registration failed: No valid face images found.\n" + "\n".join(reasons),
            )

        # Compute aggregate mean embedding
        mean_vector = np.mean(valid_embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vector)
        if norm > 1e-6:
            mean_vector = mean_vector / norm

        record = EmployeeRecord(
            employee_id=clean_id,
            employee_name=clean_name,
            registered_at=datetime.now().isoformat(),
            sample_count=valid_count,
            mean_embedding=mean_vector,
            sample_embeddings=valid_embeddings,
            metadata=metadata or {},
        )

        # Save to storage (NO RAW IMAGES STORED)
        self.storage.save_record(record)

        msg = f"Successfully registered '{clean_name}' (ID: {clean_id}) using {valid_count} sample(s)."
        if rejected_count > 0:
            msg += f" Note: {rejected_count} ambiguous/invalid sample(s) were rejected."

        return RegistrationResult(
            success=True,
            employee_id=clean_id,
            employee_name=clean_name,
            valid_samples=valid_count,
            rejected_samples=rejected_count,
            evaluations=evaluations,
            message=msg,
            record=record,
        )

    # -----------------------------------------------------------------------
    # Recognition / Identification
    # -----------------------------------------------------------------------

    # Minimum gap between best and second-best match to accept identification.
    # Prevents ambiguous matches where two employees score almost equally.
    MATCH_MARGIN: float = 0.09

    def registered_employee_ids(self) -> Set[str]:
        """Return the IDs of all currently enrolled employees."""
        return {rec.employee_id for rec in self.storage.load_all_records()}

    def identify(
        self,
        face_input: Union[np.ndarray, str, Path],
        threshold: Optional[float] = None,
        prefer_center: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Identify a person by matching their face against the registered employee database.

        Strictly enforces the recognition threshold: does NOT automatically identify
        unknown individuals.

        Args:
            face_input: Face embedding array (128,) or cropped BGR face image.
            threshold: Cosine similarity threshold (defaults to self.default_threshold).
            prefer_center: When multiple faces are detected, select face closest to horizontal center.

        Returns:
            Dictionary with employee_id, employee_name, similarity if matched, or None if unknown.
        """
        thresh = threshold if threshold is not None else self.default_threshold

        # 1. Obtain query embedding
        if isinstance(face_input, np.ndarray) and face_input.ndim == 1 and face_input.shape[0] == 128:
            query_emb = face_input
        else:
            # Assume image input
            if isinstance(face_input, (str, Path)):
                img = cv2.imread(str(face_input))
            else:
                img = face_input

            if img is None or img.size == 0:
                return None

            faces = self.engine.detect_faces(img)
            if not faces:
                return None

            img_w = img.shape[1]
            img_cx = img_w / 2.0

            if prefer_center:
                # Reject faces whose center falls in the outer 15% of the crop
                # horizontally — these likely belong to adjacent persons whose
                # face bleeds into this person's bounding-box crop.
                edge_margin = 0.15
                valid_faces = [
                    f for f in faces
                    if edge_margin <= (f.x + f.w / 2.0) / img_w <= (1.0 - edge_margin)
                ]
                if not valid_faces:
                    return None

                # Among valid faces, pick the one closest to horizontal center
                target_face = min(valid_faces, key=lambda f: abs((f.x + f.w / 2.0) - img_cx))
            else:
                target_face = faces[0]

            query_emb = self.engine.compute_embedding(img, target_face)

        # 2. Compare against all registered employees using ONLY the mean
        # embedding.  Using individual sample embeddings inflates scores via
        # max-of-N and causes false matches across different people.
        records = self.storage.load_all_records()
        if not records:
            return None

        scored: list = []  # [(similarity, record), ...]
        for rec in records:
            sim = self.engine.compute_similarity(query_emb, rec.mean_embedding)
            scored.append((sim, rec))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[0], reverse=True)

        best_similarity, best_match = scored[0]

        # 3. Check threshold
        if best_similarity < thresh:
            return None

        # 4. Margin check — the best must be clearly better than the runner-up.
        # This prevents ambiguous identifications where two employees score
        # almost equally (e.g. 0.42 vs 0.40), which is indistinguishable noise.
        if len(scored) > 1:
            second_similarity = scored[1][0]
            margin = best_similarity - second_similarity
            if margin < self.MATCH_MARGIN:
                logger.debug(
                    "Ambiguous match rejected: best=%.4f (%s) vs second=%.4f (%s), margin=%.4f < %.4f",
                    best_similarity, best_match.employee_id,
                    second_similarity, scored[1][1].employee_id,
                    margin, self.MATCH_MARGIN,
                )
                return None

        return {
            "employee_id": best_match.employee_id,
            "employee_name": best_match.employee_name,
            "similarity": best_similarity,
            "threshold": thresh,
            "matched": True,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for employee registration."""
    parser = argparse.ArgumentParser(
        description="Register an employee with multiple face images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--id", type=str, required=True, help="Unique employee ID (e.g. EMP001).")
    parser.add_argument("--name", type=str, required=True, help="Employee full name.")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to face images for enrollment.")
    parser.add_argument("--storage-dir", type=str, default=str(DEFAULT_STORAGE_DIR), help="Directory for database.")
    parser.add_argument("--overwrite", action="store_true", help="Allow updating an existing employee record.")
    return parser


def main() -> None:
    """CLI entrypoint for employee registration."""
    parser = build_parser()
    args = parser.parse_args()

    storage = EmployeeStorage(storage_dir=args.storage_dir)
    service = EmployeeRegistrationService(storage=storage)

    result = service.register_employee(
        employee_id=args.id,
        employee_name=args.name,
        images=args.images,
        allow_overwrite=args.overwrite,
    )

    print("\n" + "=" * 60)
    if result.success:
        print(f"  SUCCESS: {result.message}")
        print(f"  Enrolled ID     : {result.employee_id}")
        print(f"  Employee Name   : {result.employee_name}")
        print(f"  Valid Samples   : {result.valid_samples}")
        print(f"  Rejected Samples: {result.rejected_samples}")
    else:
        print(f"  FAILED: {result.message}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
