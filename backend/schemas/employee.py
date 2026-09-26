"""Employee registration and management schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmployeeResponse(BaseModel):
    """Schema for a registered employee record."""

    employee_id: str = Field(..., description="Unique employee identifier")
    employee_name: str = Field(..., description="Full employee name")
    registered_at: str = Field(..., description="Registration timestamp in ISO 8601 format")
    sample_count: int = Field(..., description="Number of enrolled face samples")
    embedding_dim: int = Field(128, description="Dimensionality of face embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata dictionary")


class EmployeeListResponse(BaseModel):
    """Schema for GET /employees listing response."""

    total: int = Field(..., description="Total enrolled employees")
    employees: List[EmployeeResponse] = Field(default_factory=list, description="List of employee records")


class EmployeeRegistrationResponse(BaseModel):
    """Schema for POST /employees registration response."""

    success: bool = Field(..., description="Whether registration was successful")
    employee_id: str = Field(..., description="Enrolled employee ID")
    employee_name: str = Field(..., description="Enrolled employee name")
    valid_samples: int = Field(..., description="Number of valid face samples processed")
    rejected_samples: int = Field(..., description="Number of rejected samples (no face or multiple faces)")
    message: str = Field(..., description="Result message or failure reason")
