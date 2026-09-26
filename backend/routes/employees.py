"""Employee enrollment and database API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from backend.schemas.employee import (
    EmployeeListResponse,
    EmployeeRegistrationResponse,
    EmployeeResponse,
)
from backend.services.employee_service import EmployeeService

router = APIRouter(prefix="/employees", tags=["Employees"])


def get_employee_service(request: Request) -> EmployeeService:
    """Helper to retrieve the EmployeeService from application state."""
    service: Optional[EmployeeService] = getattr(request.app.state, "employee_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Employee service is not initialized.",
        )
    return service


@router.get(
    "",
    response_model=EmployeeListResponse,
    summary="List all registered employees",
)
async def list_employees(request: Request) -> EmployeeListResponse:
    """Retrieve catalog of all registered employees and sample counts."""
    service = get_employee_service(request)
    return service.list_employees()


@router.post(
    "",
    response_model=EmployeeRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new employee with face images",
)
async def register_employee(
    request: Request,
    employee_id: Optional[str] = Form(None, description="Unique employee identifier (e.g. 'EMP001')"),
    employee_name: Optional[str] = Form(None, description="Full employee name (e.g. 'Jane Doe')"),
    allow_overwrite: bool = Form(False, description="Whether to overwrite existing employee record"),
    images: Optional[List[UploadFile]] = File(None, description="One or more face image files (JPEG, PNG)"),
) -> EmployeeRegistrationResponse:
    """Enroll a new employee using face images.

    Extracts facial features using YuNet and SFace ONNX models, validates single face
    presence, and persists 128-D normalized embeddings without retaining raw images.
    """
    service = get_employee_service(request)

    clean_id = (employee_id or "").strip()
    clean_name = (employee_name or "").strip()

    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee ID cannot be empty or whitespace.",
        )

    if not clean_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee name cannot be empty or whitespace.",
        )

    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one face image must be provided.",
        )

    # Read uploaded image bytes
    raw_images: List[bytes] = []
    for upload in images:
        content = await upload.read()
        if content:
            raw_images.append(content)

    if not raw_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image files were empty.",
        )

    response = service.register_employee(
        employee_id=clean_id,
        employee_name=clean_name,
        image_bytes_list=raw_images,
        allow_overwrite=allow_overwrite,
    )

    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.message,
        )

    return response


@router.get(
    "/{employee_id}",
    response_model=EmployeeResponse,
    summary="Get single employee details",
)
async def get_employee(
    employee_id: str,
    request: Request,
) -> EmployeeResponse:
    """Look up an employee's enrollment metadata by ID."""
    service = get_employee_service(request)
    emp = service.get_employee(employee_id)
    if emp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee '{employee_id}' not found.",
        )
    return emp


@router.delete(
    "/{employee_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an employee record",
)
async def delete_employee(
    employee_id: str,
    request: Request,
) -> dict:
    """Delete an employee and their stored embeddings."""
    service = get_employee_service(request)
    deleted = service.delete_employee(employee_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee '{employee_id}' not found.",
        )
    return {"detail": f"Employee '{employee_id}' deleted successfully."}
