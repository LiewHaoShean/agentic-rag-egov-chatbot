"""GET /health — liveness + Redis reachability."""
from fastapi import APIRouter

from core.redis_client import ping
from models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", redis=ping())
