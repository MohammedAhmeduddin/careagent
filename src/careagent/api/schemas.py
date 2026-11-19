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


# ── Response ───────────────────────────────────────────────────────────────────

class MetricComparison(BaseModel):
    label: str
    ratio_to_peer_median: Optional[float] = Field(
        None, description="Provider value / specialty median (risk-adjusted where noted)")
    percentile: Optional[float] = Field(None, description="Within-specialty percentile, 0-100")
    modified_z: Optional[float] = Field(None, description="Iglewicz-Hoaglin modified z-score")


class ScorecardResponse(BaseModel):
    npi: str
    run_id: str
    provider_type: Optional[str]
    city: Optional[str]
    state: Optional[str]

    billing_pattern_status: Optional[str] = Field(
        None, description="typical | outlier | insufficient_peers | insufficient_data")
    peer_group_size: Optional[int]
    provider_metrics: Optional[dict]
    peer_benchmarks: Optional[dict]
    peer_comparison: Optional[dict[str, MetricComparison]]

    is_outlier: Optional[bool]
    outlier_reasons: list[str]
    isolation_forest_percentile: Optional[float] = Field(
        None, description="Multivariate unusualness within the specialty; 100 = most unusual")

    data_completeness: Optional[float]
    missing_fields: list[str]

    performance_narrative: Optional[str]
    narrative_source: Optional[str] = Field(
        None, description='"llm" if the model wrote it (and it passed the number check), else "template"')
    disclaimer: str

    scorecard_version: Optional[str]
    agents_executed: list[str]
    agents_skipped: list[str]
    pipeline_duration_seconds: Optional[float]
    generated_at: datetime


class HealthResponse(BaseModel):
    status: str
    database: str
    providers_loaded: int
    providers_compared: int
    outliers_flagged: int
    peer_groups: int
    version: str = "2.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    npi: Optional[str] = None
