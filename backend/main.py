"""
FastAPI Backend Application for Real-Time PPE Compliance Monitoring.

Exposes REST and WebSocket endpoints for:
  - System diagnostics (GET /health)
  - Employee face enrollment (GET /employees, POST /employees)
  - PPE monitoring telemetry and status (GET /monitoring/status, WebSocket /ws/monitoring)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.routes.employees import router as employees_router
from backend.routes.health import router as health_router
from backend.routes.monitoring import router as monitoring_router
from backend.services.employee_service import EmployeeService
from backend.services.monitoring_service import MonitoringService
from detection.pipeline import PPECompliancePipeline
from recognition.registration import EmployeeRegistrationService

logger = logging.getLogger("backend")


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown lifecycles of backend services."""
    # 1. Startup
    logger.info("Initializing PPE Compliance Backend...")
    app.state.start_time = time.time()

    loop = asyncio.get_running_loop()

    # Initialize business services (only if not already injected by tests)
    if not hasattr(app.state, "employee_service") or app.state.employee_service is None:
        app.state.employee_service = EmployeeService()

    if not hasattr(app.state, "monitoring_service") or app.state.monitoring_service is None:
        shared_storage = getattr(app.state.employee_service, "storage", None)
        pipeline_face_service = EmployeeRegistrationService(storage=shared_storage)
        pipeline = PPECompliancePipeline(face_service=pipeline_face_service)
        app.state.monitoring_service = MonitoringService(pipeline=pipeline, event_loop=loop)
    else:
        app.state.monitoring_service.set_event_loop(loop)

    logger.info("PPE Compliance Backend successfully initialized.")

    yield

    # 2. Shutdown
    logger.info("Shutting down PPE Compliance Backend...")
    if hasattr(app.state, "monitoring_service") and app.state.monitoring_service:
        app.state.monitoring_service.shutdown()
    logger.info("PPE Compliance Backend shutdown complete.")


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------


def create_app(
    employee_service: Optional[EmployeeService] = None,
    monitoring_service: Optional[MonitoringService] = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Allows injecting mock services for deterministic unit testing.
    """
    app = FastAPI(
        title="Real-Time PPE Compliance & Identity Monitoring API",
        description=(
            "Backend API providing automated PPE safety equipment detection, "
            "employee identity verification, and real-time WebSocket telemetry."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Injected services for tests
    if employee_service is not None:
        app.state.employee_service = employee_service
    if monitoring_service is not None:
        app.state.monitoring_service = monitoring_service

    # -----------------------------------------------------------------------
    # CORS Configuration
    # -----------------------------------------------------------------------
    cors_origins_env = os.getenv("CORS_ORIGINS", "*")
    origins: List[str] = (
        [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        if cors_origins_env != "*"
        else ["*"]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Exception Handlers
    # -----------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Request validation failed", "errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    # -----------------------------------------------------------------------
    # Routers
    # -----------------------------------------------------------------------
    app.include_router(health_router)
    app.include_router(employees_router)
    app.include_router(monitoring_router)

    return app


# Default ASGI application instance
app = create_app()


# ---------------------------------------------------------------------------
# Direct CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting Uvicorn server on http://%s:%d", host, port)
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
