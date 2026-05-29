"""
careagent.api.schemas
~~~~~~~~~~~~~~~~~~~~~~
Pydantic v2 request/response schemas for the FastAPI endpoints.
These are the typed contracts between the API and its callers.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    npi: str = Field(
        ...,
        min_length=10,
        max_length=10,
        pattern=r"^\d{10}$",
        description="10-digit National Provider Identifier",
        examples=["1234567890"],
    )
    force_rerun: bool = Field(
        default=False,
        description="Re-run pipeline even if provider was scored recently",
    )


# ── Response ───────────────────────────────────────────────────────────────────

class ScorecardResponse(BaseModel):
    npi: str
    run_id: str
    provider_name: Optional[str]
    provider_type: Optional[str]
    state: Optional[str]

    # Quality metrics
    quality_score: Optional[float]
    quality_percentile: Optional[float]
    cost_efficiency_ratio: Optional[float]
    volume_percentile: Optional[float]

    # Anomaly
    is_anomaly: Optional[bool]
    anomaly_score: Optional[float]
    anomaly_reason: Optional[str]

    # Data quality
    data_quality_score: Optional[float]
    fields_imputed: Optional[int]

    # Narrative
    performance_narrative: Optional[str]
    narrative_faithfulness: Optional[float]
    narrative_relevancy: Optional[float]

    # Recommendation
    network_recommendation: Optional[str]
    scorecard_version: Optional[str]

    # Pipeline metadata
    agents_executed: list[str]
    agents_skipped: list[str]
    pipeline_duration_seconds: Optional[float]
    generated_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    database: str
    providers_loaded: int
    providers_scored: int
    anomalies_flagged: int
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    npi: Optional[str] = None
