"""Face recognition and employee registration package."""

from recognition.embeddings import FaceDetection, FaceEmbeddingEngine
from recognition.registration import (
    EmployeeRecord,
    EmployeeRegistrationService,
    EmployeeStorage,
    RegistrationResult,
    RegistrationStatus,
)

__all__ = [
    "EmployeeRecord",
    "EmployeeRegistrationService",
    "EmployeeStorage",
    "FaceDetection",
    "FaceEmbeddingEngine",
    "RegistrationResult",
    "RegistrationStatus",
]
