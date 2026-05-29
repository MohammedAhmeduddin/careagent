"""
careagent.db.queries
~~~~~~~~~~~~~~~~~~~~
All typed SQL queries for CareAgent agents.
"""

from datetime import datetime, UTC
from typing import Optional

from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from careagent.db.models import AgentRun, Provider, ProviderService


def get_provider_by_npi(db: Session, npi: str) -> Optional[Provider]:
    return db.get(Provider, npi)


def get_providers_for_scoring(
    db: Session,
    state: Optional[str] = None,
    specialty: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Provider]:
    stmt = (
        select(Provider)
        .where(Provider.quality_score.is_(None))
        .order_by(Provider.npi)
        .limit(limit)
        .offset(offset)
    )
    if state:
        stmt = stmt.where(Provider.state == state)
    if specialty:
        stmt = stmt.where(Provider.provider_type == specialty)
    return list(db.scalars(stmt).all())


def get_provider_services(db: Session, npi: str) -> list[ProviderService]:
    stmt = select(ProviderService).where(ProviderService.npi == npi)
    return list(db.scalars(stmt).all())


def get_specialty_benchmarks(db: Session, provider_type: str) -> dict:
    stmt = select(
        func.avg(Provider.avg_medicare_payment).label("avg_payment"),
        func.avg(Provider.avg_submitted_charge).label("avg_charge"),
        func.avg(Provider.total_services).label("avg_services"),
        func.avg(Provider.total_unique_beneficiaries).label("avg_beneficiaries"),
        func.count(Provider.npi).label("provider_count"),
    ).where(Provider.provider_type == provider_type)

    row = db.execute(stmt).mappings().first()
    if not row or not row["provider_count"]:
        logger.warning(f"No benchmark data found for specialty: {provider_type}")
        return {}

    return {
        "avg_payment":       row["avg_payment"],
        "avg_charge":        row["avg_charge"],
        "avg_services":      row["avg_services"],
        "avg_beneficiaries": row["avg_beneficiaries"],
        "provider_count":    row["provider_count"],
    }


def get_all_providers_for_anomaly_detection(
    db: Session, min_services: float = 10.0
) -> list[Provider]:
    stmt = (
        select(Provider)
        .where(Provider.total_services >= min_services)
        .where(Provider.avg_medicare_payment.is_not(None))
        .order_by(Provider.npi)
    )
    return list(db.scalars(stmt).all())


def update_provider_quality_scores(
    db: Session, npi: str, quality_score: float,
    quality_percentile: float, cost_efficiency_ratio: float,
    volume_percentile: float,
) -> None:
    db.execute(
        update(Provider).where(Provider.npi == npi).values(
            quality_score=quality_score,
            quality_percentile=quality_percentile,
            cost_efficiency_ratio=cost_efficiency_ratio,
            volume_percentile=volume_percentile,
        )
    )


def update_provider_anomaly(
    db: Session, npi: str, is_anomaly: bool,
    anomaly_score: float, anomaly_reason: str,
) -> None:
    db.execute(
        update(Provider).where(Provider.npi == npi).values(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_reason=anomaly_reason,
        )
    )


def update_provider_narrative(
    db: Session, npi: str, narrative: str,
    faithfulness: Optional[float], relevancy: Optional[float],
) -> None:
    db.execute(
        update(Provider).where(Provider.npi == npi).values(
            performance_narrative=narrative,
            narrative_faithfulness=faithfulness,
            narrative_relevancy=relevancy,
        )
    )


def update_provider_cleaning(
    db: Session, npi: str, data_quality_score: float,
    fields_imputed: int, cleaning_notes: str,
) -> None:
    db.execute(
        update(Provider).where(Provider.npi == npi).values(
            data_quality_score=data_quality_score,
            fields_imputed=fields_imputed,
            cleaning_notes=cleaning_notes,
        )
    )


def update_provider_recommendation(
    db: Session, npi: str, recommendation: str, scorecard_version: str,
) -> None:
    db.execute(
        update(Provider).where(Provider.npi == npi).values(
            network_recommendation=recommendation,
            scorecard_version=scorecard_version,
            last_scored_at=datetime.now(UTC),
        )
    )


def create_agent_run(db: Session, run_id: str, npi: str) -> AgentRun:
    run = AgentRun(run_id=run_id, npi=npi, status="running")
    db.add(run)
    db.flush()
    return run


def complete_agent_run(
    db: Session, run_id: str, agents_executed: list[str],
    agents_skipped: list[str], duration_seconds: float,
    tokens_used: int, llm_calls: int, recommendation: str,
    mlflow_run_id: Optional[str] = None,
) -> None:
    db.execute(
        update(AgentRun).where(AgentRun.run_id == run_id).values(
            status="completed",
            agents_executed=",".join(agents_executed),
            agents_skipped=",".join(agents_skipped),
            total_duration_seconds=duration_seconds,
            llm_tokens_used=tokens_used,
            llm_calls_made=llm_calls,
            final_recommendation=recommendation,
            mlflow_run_id=mlflow_run_id,
            completed_at=datetime.now(UTC),
        )
    )


def fail_agent_run(db: Session, run_id: str, error_message: str) -> None:
    db.execute(
        update(AgentRun).where(AgentRun.run_id == run_id).values(
            status="failed",
            error_message=error_message[:2000],
            completed_at=datetime.now(UTC),
        )
    )


def get_database_stats(db: Session) -> dict:
    return {
        "total_providers":   db.scalar(select(func.count(Provider.npi))) or 0,
        "providers_scored":  db.scalar(select(func.count(Provider.npi)).where(Provider.quality_score.is_not(None))) or 0,
        "anomalies_flagged": db.scalar(select(func.count(Provider.npi)).where(Provider.is_anomaly.is_(True))) or 0,
        "total_agent_runs":  db.scalar(select(func.count(AgentRun.id))) or 0,
        "completed_runs":    db.scalar(select(func.count(AgentRun.id)).where(AgentRun.status == "completed")) or 0,
    }
