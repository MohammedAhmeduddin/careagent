"""
careagent.agents.anomaly
~~~~~~~~~~~~~~~~~~~~~~~~~
Anomaly agent: is this provider's billing pattern an outlier among
same-specialty peers, and why?

The flag rule and the per-specialty Isolation Forest are fitted once at
load time (careagent.scoring.peer), not per request, so every provider is
judged against the same fitted model and requests stay fast. Reasons are
the metrics that crossed the rule, with their actual ratios, not
hard-coded text.
"""

from loguru import logger

from careagent.db.queries import get_provider_by_npi
from careagent.db.session import get_db
from careagent.graph.state import AgentState
from careagent.scoring.peer import STATUS_OUTLIER


def anomaly_agent(state: AgentState) -> AgentState:
    npi = state["npi"]
    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}
        is_outlier = provider.billing_pattern_status == STATUS_OUTLIER
        reasons = [r for r in (provider.outlier_reasons or "").split("; ") if r]
        if_pct = provider.isolation_forest_percentile

    logger.info(f"[Anomaly] NPI={npi} outlier={is_outlier} reasons={len(reasons)}")
    return {
        **state,
        "is_outlier": is_outlier,
        "outlier_reasons": reasons,
        "isolation_forest_percentile": if_pct,
        "anomaly_complete": True,
        "agents_executed": state.get("agents_executed", []) + ["anomaly"],
    }
