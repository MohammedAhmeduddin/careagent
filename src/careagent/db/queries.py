"""
careagent.db.queries
~~~~~~~~~~~~~~~~~~~~
Typed SQL queries used by the agents, the API and the loader.
"""

from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.orm import Session

from careagent.db.models import AgentRun, PeerGroup, Provider
from careagent.scoring.peer import STATUS_OUTLIER, STATUS_TYPICAL


def get_provider_by_npi(db: Session, npi: str) -> Optional[Provider]:
    return db.get(Provider, npi)


def get_peer_group(db: Session, provider_type: Optional[str]) -> Optional[PeerGroup]:
    if not provider_type:
        return None
    return db.get(PeerGroup, provider_type)


def get_example_npis(db: Session) -> dict[str, Optional[str]]:
    """One NPI per status, for the demo. Deterministic (lowest NPI)."""
    examples = {}
    for status in (STATUS_TYPICAL, STATUS_OUTLIER, "insufficient_peers"):
        examples[status] = db.scalar(
            select(Provider.npi)
            .where(Provider.billing_pattern_status == status)
            .order_by(Provider.npi)
            .limit(1)
        )
    return examples


def update_provider_validation(db: Session, npi: str, notes: str) -> None:
    db.execute(update(Provider).where(Provider.npi == npi).values(validation_notes=notes))


def update_provider_narrative(
    db: Session, npi: str, narrative: str, source: str, fingerprint: Optional[str] = None,
) -> None:
    db.execute(
        update(Provider).where(Provider.npi == npi).values(
            performance_narrative=narrative, narrative_source=source,
            narrative_fingerprint=fingerprint,
        )
    )


def get_cached_narrative(db: Session, npi: str, fingerprint: str) -> Optional[str]:
    """A stored LLM narrative, if it was written for exactly these facts."""
    row = db.execute(
        select(Provider.performance_narrative)
        .where(Provider.npi == npi)
        .where(Provider.narrative_source == "llm")
        .where(Provider.narrative_fingerprint == fingerprint)
    ).first()
    return row[0] if row and row[0] else None


def update_provider_report(db: Session, npi: str, scorecard_version: str) -> None:
    db.execute(
        update(Provider).where(Provider.npi == npi).values(
            scorecard_version=scorecard_version, last_scored_at=datetime.now(UTC),
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
    tokens_used: int, llm_calls: int, final_status: Optional[str],
) -> None:
    db.execute(
        update(AgentRun).where(AgentRun.run_id == run_id).values(
            status="completed",
            agents_executed=",".join(agents_executed),
            agents_skipped=",".join(agents_skipped),
            total_duration_seconds=duration_seconds,
            llm_tokens_used=tokens_used,
            llm_calls_made=llm_calls,
            final_status=final_status,
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
    def count_status(status: str) -> int:
        return db.scalar(
            select(func.count(Provider.npi)).where(Provider.billing_pattern_status == status)
        ) or 0

    return {
        "total_providers": db.scalar(select(func.count(Provider.npi))) or 0,
        "providers_compared": count_status(STATUS_TYPICAL) + count_status(STATUS_OUTLIER),
        "outliers_flagged": count_status(STATUS_OUTLIER),
        "peer_groups": db.scalar(select(func.count(PeerGroup.provider_type))) or 0,
        "total_agent_runs": db.scalar(select(func.count(AgentRun.id))) or 0,
    }


def replace_all_data(
    db: Session, providers: list[dict], peer_groups: list[dict], chunk_size: int = 5000
) -> None:
    """Replace providers and peer groups in one transaction (agent runs go too)."""
    db.execute(delete(AgentRun))
    db.execute(delete(Provider))
    db.execute(delete(PeerGroup))
    if peer_groups:
        db.execute(insert(PeerGroup), peer_groups)
    for start in range(0, len(providers), chunk_size):
        db.execute(insert(Provider), providers[start:start + chunk_size])
