"""
careagent.api.routes.health
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Health check endpoint. Used by Docker, GCP Cloud Run, and the dashboard.
"""

from fastapi import APIRouter
from careagent.api.schemas import HealthResponse
from careagent.db.session import get_db, check_connection
from careagent.db.queries import get_database_stats

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    db_ok = check_connection()
    if not db_ok:
        return HealthResponse(
            status="degraded",
            database="unreachable",
            providers_loaded=0,
            providers_scored=0,
            anomalies_flagged=0,
        )
    with get_db() as db:
        stats = get_database_stats(db)
    return HealthResponse(
        status="healthy",
        database="connected",
        providers_loaded=stats["total_providers"],
        providers_scored=stats["providers_scored"],
        anomalies_flagged=stats["anomalies_flagged"],
    )
