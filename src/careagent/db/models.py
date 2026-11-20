"""
careagent.db.models
~~~~~~~~~~~~~~~~~~~
SQLAlchemy ORM models.

Source data: CMS "Medicare Physician & Other Practitioners - by Provider"
(one row per NPI). Peer comparisons are computed once at load time by
careagent.scoring.peer and stored here; agents read them per request.

Provider names and street addresses are deliberately not stored: the demo
is public, and a billing-pattern outlier is not a finding about a person.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"

    # Identity (no names)
    npi: Mapped[str] = mapped_column(String(10), primary_key=True)
    entity_type: Mapped[Optional[str]] = mapped_column(String(1))
    provider_type: Mapped[Optional[str]] = mapped_column(String(100))
    medicare_participating: Mapped[Optional[str]] = mapped_column(String(1))

    # Location
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(2))
    zip_code: Mapped[Optional[str]] = mapped_column(String(10))
    ruca_code: Mapped[Optional[str]] = mapped_column(String(10))

    # CMS totals
    total_hcpcs_codes: Mapped[Optional[float]] = mapped_column(Float)
    total_beneficiaries: Mapped[Optional[float]] = mapped_column(Float)
    total_services: Mapped[Optional[float]] = mapped_column(Float)
    total_submitted_charges: Mapped[Optional[float]] = mapped_column(Float)
    total_allowed_amount: Mapped[Optional[float]] = mapped_column(Float)
    total_medicare_payment: Mapped[Optional[float]] = mapped_column(Float)
    total_standardized_payment: Mapped[Optional[float]] = mapped_column(Float)
    bene_avg_age: Mapped[Optional[float]] = mapped_column(Float)
    bene_avg_risk_score: Mapped[Optional[float]] = mapped_column(Float)

    # Derived metrics
    cost_per_beneficiary: Mapped[Optional[float]] = mapped_column(Float)
    services_per_beneficiary: Mapped[Optional[float]] = mapped_column(Float)
    charge_to_allowed_ratio: Mapped[Optional[float]] = mapped_column(Float)
    data_completeness: Mapped[Optional[float]] = mapped_column(Float)

    # Peer comparison (computed at load time)
    peer_group_size: Mapped[Optional[int]] = mapped_column(Integer)
    cost_z: Mapped[Optional[float]] = mapped_column(Float)
    cost_ratio_to_peer_median: Mapped[Optional[float]] = mapped_column(Float)
    cost_percentile: Mapped[Optional[float]] = mapped_column(Float)
    services_z: Mapped[Optional[float]] = mapped_column(Float)
    services_ratio_to_peer_median: Mapped[Optional[float]] = mapped_column(Float)
    services_percentile: Mapped[Optional[float]] = mapped_column(Float)
    charge_z: Mapped[Optional[float]] = mapped_column(Float)
    charge_ratio_to_peer_median: Mapped[Optional[float]] = mapped_column(Float)
    charge_percentile: Mapped[Optional[float]] = mapped_column(Float)
    isolation_forest_percentile: Mapped[Optional[float]] = mapped_column(Float)
    billing_pattern_status: Mapped[Optional[str]] = mapped_column(String(30))
    outlier_reasons: Mapped[Optional[str]] = mapped_column(Text)

    # Written by the agents
    validation_notes: Mapped[Optional[str]] = mapped_column(Text)
    performance_narrative: Mapped[Optional[str]] = mapped_column(Text)
    narrative_source: Mapped[Optional[str]] = mapped_column(String(20))
    # sha256 of model + prompt version + facts; an LLM narrative is reused
    # only while all three are unchanged
    narrative_fingerprint: Mapped[Optional[str]] = mapped_column(String(64))
    scorecard_version: Mapped[Optional[str]] = mapped_column(String(20))
    last_scored_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Record metadata
    cms_data_year: Mapped[int] = mapped_column(Integer, nullable=False, default=2024)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    agent_runs: Mapped[list["AgentRun"]] = relationship(
        "AgentRun", back_populates="provider", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_providers_state", "state"),
        Index("ix_providers_provider_type", "provider_type"),
        Index("ix_providers_billing_pattern_status", "billing_pattern_status"),
    )

    def __repr__(self) -> str:
        return f"<Provider npi={self.npi} type={self.provider_type}>"


class PeerGroup(Base):
    """One row per specialty that had enough providers to compare."""

    __tablename__ = "peer_groups"

    provider_type: Mapped[str] = mapped_column(String(100), primary_key=True)
    peer_count: Mapped[int] = mapped_column(Integer, nullable=False)
    outlier_count: Mapped[int] = mapped_column(Integer, nullable=False)
    median_cost_per_beneficiary: Mapped[float] = mapped_column(Float, nullable=False)
    median_services_per_beneficiary: Mapped[float] = mapped_column(Float, nullable=False)
    median_charge_to_allowed_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    median_risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    cost_risk_slope: Mapped[float] = mapped_column(Float, nullable=False)
    services_risk_slope: Mapped[float] = mapped_column(Float, nullable=False)
    cms_data_year: Mapped[int] = mapped_column(Integer, nullable=False, default=2024)

    def __repr__(self) -> str:
        return f"<PeerGroup {self.provider_type} n={self.peer_count}>"


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    npi: Mapped[str] = mapped_column(String(10), ForeignKey("providers.npi", ondelete="CASCADE"), nullable=False)

    agents_executed: Mapped[Optional[str]] = mapped_column(Text)
    agents_skipped: Mapped[Optional[str]] = mapped_column(Text)

    total_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    llm_tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    llm_calls_made: Mapped[Optional[int]] = mapped_column(Integer)

    final_status: Mapped[Optional[str]] = mapped_column(String(30))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    provider: Mapped["Provider"] = relationship("Provider", back_populates="agent_runs")

    __table_args__ = (
        Index("ix_agent_runs_npi", "npi"),
        Index("ix_agent_runs_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<AgentRun run_id={self.run_id} npi={self.npi} status={self.status}>"
