"""Routes package."""

from backend.routes.employees import router as employees_router
from backend.routes.health import router as health_router
from backend.routes.monitoring import router as monitoring_router

__all__ = ["employees_router", "health_router", "monitoring_router"]
