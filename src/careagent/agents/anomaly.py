"""
careagent.agents.anomaly
~~~~~~~~~~~~~~~~~~~~~~~~~
Anomaly Detection Agent — Isolation Forest on cost + quality metrics.
Week 2: stub. Full Isolation Forest implementation in Week 3.
"""

from loguru import logger
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.queries import get_provider_by_npi, update_provider_anomaly
from careagent.config import get_settings

settings = get_settings()


def anomaly_agent(state: AgentState) -> AgentState:
    """Flags providers with anomalous cost/quality patterns."""
    npi = state["npi"]
    logger.info(f"[Anomaly] Processing NPI={npi}")

    cost_ratio   = state.get("cost_efficiency_ratio") or 1.0
    quality      = state.get("quality_score") or 50.0

    # Stub logic: flag if cost ratio is very high AND quality is low
    # Week 3: replace with full Isolation Forest trained on all providers
    is_anomaly   = (cost_ratio > 3.5) and (quality < 40.0)
    anomaly_score = round(-1.0 * cost_ratio / 10.0, 4)  # stub score
    reason = (
        f"High cost-efficiency ratio ({cost_ratio:.2f}) with low quality score ({quality:.1f})"
        if is_anomaly else
        f"Within normal range — cost ratio {cost_ratio:.2f}, quality {quality:.1f}"
    )

    with get_db() as db:
        update_provider_anomaly(db, npi,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_reason=reason,
        )

    executed = state.get("agents_executed", []) + ["anomaly"]
    logger.info(f"[Anomaly] NPI={npi} flagged={is_anomaly}")

    return {
        **state,
        "is_anomaly":      is_anomaly,
        "anomaly_score":   anomaly_score,
        "anomaly_reason":  reason,
        "anomaly_complete": True,
        "agents_executed": executed,
    }
