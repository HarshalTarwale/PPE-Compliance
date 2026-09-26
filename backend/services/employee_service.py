"""
Employee business service.

Wraps the recognition and registration models, abstracting all disk I/O,
image decoding, and database queries from the API routes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.schemas.employee import (
    EmployeeListResponse,
    EmployeeRegistrationResponse,
    EmployeeResponse,
)
from recognition.registration import (
    EmployeeRecord,
    EmployeeRegistrationService,
    EmployeeStorage,
    RegistrationResult,
)

logger = logging.getLogger("employee_service")


class EmployeeService:
    """Business logic for employee face enrollment and identity catalog."""

    def __init__(
        self,
        registration_service: Optional[EmployeeRegistrationService] = None,
        storage: Optional[EmployeeStorage] = None,
    ) -> None:
        self.storage = storage or EmployeeStorage()
        self.reg_service = registration_service or EmployeeRegistrationService(
            storage=self.storage
        )

    def list_employees(self) -> EmployeeListResponse:
        """Retrieve metadata for all registered employees."""
        records = self.storage.list_records()
        employee_responses: List[EmployeeResponse] = []

        for rec in records:
            employee_responses.append(
                EmployeeResponse(
                    employee_id=rec["employee_id"],
                    employee_name=rec["employee_name"],
                    registered_at=rec.get("registered_at", ""),
                    sample_count=rec.get("sample_count", 0),
                    embedding_dim=rec.get("embedding_dim", 128),
                    metadata=rec.get("metadata", {}),
                )
            )

        return EmployeeListResponse(
            total=len(employee_responses),
            employees=employee_responses,
        )

    def get_employee(self, employee_id: str) -> Optional[EmployeeResponse]:
        """Fetch metadata for a specific employee."""
        rec = self.storage.load_record(employee_id.strip())
        if rec is None:
            return None

        return EmployeeResponse(
            employee_id=rec.employee_id,
            employee_name=rec.employee_name,
            registered_at=rec.registered_at,
            sample_count=rec.sample_count,
            embedding_dim=int(rec.mean_embedding.shape[0]),
            metadata=rec.metadata,
        )

    def register_employee(
        self,
        employee_id: str,
        employee_name: str,
        image_bytes_list: List[bytes],
        allow_overwrite: bool = False,
    ) -> EmployeeRegistrationResponse:
        """Process and enroll an employee from raw image bytes.

        Decodes image bytes into BGR numpy arrays, validates single face presence,
        and saves normalized embeddings without retaining raw photos.
        """
        clean_id = employee_id.strip()
        clean_name = employee_name.strip()

        if not clean_id:
            return EmployeeRegistrationResponse(
                success=False,
                employee_id="",
                employee_name=clean_name,
                valid_samples=0,
                rejected_samples=len(image_bytes_list),
                message="Employee ID cannot be empty.",
            )

        if not clean_name:
            return EmployeeRegistrationResponse(
                success=False,
                employee_id=clean_id,
                employee_name="",
                valid_samples=0,
                rejected_samples=len(image_bytes_list),
                message="Employee name cannot be empty.",
            )

        if not image_bytes_list:
            return EmployeeRegistrationResponse(
                success=False,
                employee_id=clean_id,
                employee_name=clean_name,
                valid_samples=0,
                rejected_samples=0,
                message="No face images provided for registration.",
            )

        # Decode raw bytes into OpenCV BGR numpy arrays
        decoded_images: List[np.ndarray] = []
        for raw_bytes in image_bytes_list:
            if not raw_bytes:
                continue
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                decoded_images.append(img)

        if not decoded_images:
            return EmployeeRegistrationResponse(
                success=False,
                employee_id=clean_id,
                employee_name=clean_name,
                valid_samples=0,
                rejected_samples=len(image_bytes_list),
                message="Could not decode any of the uploaded image files.",
            )

        result: RegistrationResult = self.reg_service.register_employee(
            employee_id=clean_id,
            employee_name=clean_name,
            images=decoded_images,
            allow_overwrite=allow_overwrite,
        )

        return EmployeeRegistrationResponse(
            success=result.success,
            employee_id=result.employee_id,
            employee_name=result.employee_name,
            valid_samples=result.valid_samples,
            rejected_samples=result.rejected_samples,
            message=result.message,
        )

    def delete_employee(self, employee_id: str) -> bool:
        """Remove an employee from the registration database."""
        return self.storage.delete_record(employee_id.strip())
