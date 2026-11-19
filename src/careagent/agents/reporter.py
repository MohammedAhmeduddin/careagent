"""
careagent.agents.reporter
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reporter agent: assembles the final scorecard.

The outcome is a billing-pattern status (typical / outlier /
insufficient_peers / insufficient_data), not a network decision: Medicare
billing data alone cannot justify including or excluding a provider.
Every scorecard carries the disclaimer.
"""

from datetime import UTC, datetime

from loguru import logger

from careagent.db.queries import update_provider_report
from careagent.db.session import get_db
from careagent.graph.state import AgentState
from careagent.scoring.peer import DISCLAIMER

SCORECARD_VERSION = "2.0.0"


def reporter_agent(state: AgentState) -> AgentState:
    npi = state["npi"]
    scorecard = {
        "npi": npi,
        "run_id": state["run_id"],
        "provider_type": state.get("provider_type"),
        "city": state.get("city"),
        "state": state.get("state"),
        "billing_pattern_status": state.get("billing_pattern_status"),
        "peer_group_size": state.get("peer_group_size"),
        "provider_metrics": state.get("provider_metrics"),
        "peer_benchmarks": state.get("peer_benchmarks"),
        "peer_comparison": state.get("peer_comparison"),
        "is_outlier": state.get("is_outlier"),
        "outlier_reasons": state.get("outlier_reasons") or [],
        "isolation_forest_percentile": state.get("isolation_forest_percentile"),
        "data_completeness": state.get("data_completeness"),
        "performance_narrative": state.get("performance_narrative"),
        "narrative_source": state.get("narrative_source"),
        "disclaimer": DISCLAIMER,
        "scorecard_version": SCORECARD_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    with get_db() as db:
        update_provider_report(db, npi, scorecard_version=SCORECARD_VERSION)

    logger.info(f"[Reporter] NPI={npi} status={scorecard['billing_pattern_status']}")
    return {
        **state,
        "scorecard": scorecard,
        "scorecard_version": SCORECARD_VERSION,
        "pipeline_complete": True,
        "agents_executed": state.get("agents_executed", []) + ["reporter"],
    }
