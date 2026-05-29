"""
careagent.db.models
~~~~~~~~~~~~~~~~~~~
SQLAlchemy ORM models for the CareAgent provider data warehouse.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"

    # Identity
    npi: Mapped[str] = mapped_column(String(10), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(1), nullable=False)
    last_name_or_org: Mapped[str] = mapped_column(String(100), nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(50))
    credentials: Mapped[Optional[str]] = mapped_column(String(50))
    gender: Mapped[Optional[str]] = mapped_column(String(1))
    provider_type: Mapped[str] = mapped_column(String(100), nullable=False)
    medicare_participation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Location
    street: Mapped[Optional[str]] = mapped_column(String(200))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    state: Mapped[Optional[str]] = mapped_column(String(2))
    zip_code: Mapped[Optional[str]] = mapped_column(String(10))
    country: Mapped[Optional[str]] = mapped_column(String(2), default="US")

    # Volume metrics
    total_services: Mapped[Optional[float]] = mapped_column(Float)
    total_unique_beneficiaries: Mapped[Optional[float]] = mapped_column(Float)
    distinct_procedure_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Cost metrics
    avg_medicare_payment: Mapped[Optional[float]] = mapped_column(Float)
    avg_submitted_charge: Mapped[Optional[float]] = mapped_column(Float)
    avg_allowed_amount: Mapped[Optional[float]] = mapped_column(Float)

    # Quality fields (populated by Statistical Agent)
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    quality_percentile: Mapped[Optional[float]] = mapped_column(Float)
    cost_efficiency_ratio: Mapped[Optional[float]] = mapped_column(Float)
    volume_percentile: Mapped[Optional[float]] = mapped_column(Float)

    # Anomaly fields (populated by Anomaly Agent)
    is_anomaly: Mapped[Optional[bool]] = mapped_column(Boolean)
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)
    anomaly_reason: Mapped[Optional[str]] = mapped_column(Text)

    # Data quality fields (populated by Data Cleaner Agent)
    data_quality_score: Mapped[Optional[float]] = mapped_column(Float)
    fields_imputed: Mapped[Optional[int]] = mapped_column(Integer)
    cleaning_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Narrative (populated by Summarizer Agent)
    performance_narrative: Mapped[Optional[str]] = mapped_column(Text)
    narrative_faithfulness: Mapped[Optional[float]] = mapped_column(Float)
    narrative_relevancy: Mapped[Optional[float]] = mapped_column(Float)

    # Scorecard (populated by Reporter Agent)
    network_recommendation: Mapped[Optional[str]] = mapped_column(String(50))
    scorecard_version: Mapped[Optional[str]] = mapped_column(String(20))
    last_scored_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Record metadata
    cms_data_year: Mapped[int] = mapped_column(Integer, nullable=False, default=2022)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    services: Mapped[list["ProviderService"]] = relationship(
        "ProviderService", back_populates="provider", cascade="all, delete-orphan"
    )
    agent_runs: Mapped[list["AgentRun"]] = relationship(
        "AgentRun", back_populates="provider"
    )

    __table_args__ = (
        Index("ix_providers_state", "state"),
        Index("ix_providers_provider_type", "provider_type"),
        Index("ix_providers_is_anomaly", "is_anomaly"),
        Index("ix_providers_network_recommendation", "network_recommendation"),
    )

    def __repr__(self) -> str:
        return f"<Provider npi={self.npi} name={self.last_name_or_org}>"


class ProviderService(Base):
    __tablename__ = "provider_services"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    npi: Mapped[str] = mapped_column(String(10), ForeignKey("providers.npi", ondelete="CASCADE"), nullable=False)
    hcpcs_code: Mapped[str] = mapped_column(String(10), nullable=False)
    hcpcs_description: Mapped[Optional[str]] = mapped_column(String(256))
    place_of_service: Mapped[Optional[str]] = mapped_column(String(1))
    is_drug_indicator: Mapped[Optional[bool]] = mapped_column(Boolean)

    line_service_count: Mapped[Optional[float]] = mapped_column(Float)
    beneficiary_unique_count: Mapped[Optional[float]] = mapped_column(Float)
    beneficiary_day_service_count: Mapped[Optional[float]] = mapped_column(Float)
    avg_medicare_allowed_amt: Mapped[Optional[float]] = mapped_column(Float)
    avg_submitted_charge_amt: Mapped[Optional[float]] = mapped_column(Float)
    avg_medicare_payment_amt: Mapped[Optional[float]] = mapped_column(Float)
    avg_medicare_standardized_amt: Mapped[Optional[float]] = mapped_column(Float)

    cms_data_year: Mapped[int] = mapped_column(Integer, nullable=False, default=2022)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    provider: Mapped["Provider"] = relationship("Provider", back_populates="services")

    __table_args__ = (
        Index("ix_services_npi", "npi"),
        Index("ix_services_hcpcs_code", "hcpcs_code"),
    )

    def __repr__(self) -> str:
        return f"<ProviderService npi={self.npi} hcpcs={self.hcpcs_code}>"


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    npi: Mapped[str] = mapped_column(String(10), ForeignKey("providers.npi"), nullable=False)

    agents_executed: Mapped[Optional[str]] = mapped_column(Text)
    agents_skipped: Mapped[Optional[str]] = mapped_column(Text)

    total_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    llm_tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    llm_calls_made: Mapped[Optional[int]] = mapped_column(Integer)

    final_recommendation: Mapped[Optional[str]] = mapped_column(String(50))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(64))

    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    provider: Mapped["Provider"] = relationship("Provider", back_populates="agent_runs")

    __table_args__ = (
        Index("ix_agent_runs_npi", "npi"),
        Index("ix_agent_runs_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<AgentRun run_id={self.run_id} npi={self.npi} status={self.status}>"
