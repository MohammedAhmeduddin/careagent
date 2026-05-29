"""
careagent.agents.reporter
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reporter Agent — assembles final structured scorecard.
Week 2: stub. Full scorecard formatting in Week 4.
"""

from datetime import datetime, UTC
from loguru import logger
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.queries import get_provider_by_npi, update_provider_recommendation
from careagent.config import get_settings

settings = get_settings()
SCORECARD_VERSION = "0.1.0"


def reporter_agent(state: AgentState) -> AgentState:
    """Assembles the final scorecard and sets network recommendation."""
    npi = state["npi"]
    logger.info(f"[Reporter] Processing NPI={npi}")

    quality    = state.get("quality_score") or 0.0
    is_anomaly = state.get("is_anomaly") or False

    # Recommendation logic
    if is_anomaly or quality < settings.quality_score_review_threshold:
        recommendation = "review"
    elif quality >= 75.0:
        recommendation = "include"
    else:
        recommendation = "review"

    scorecard = {
        "npi":                    npi,
        "run_id":                 state["run_id"],
        "quality_score":          quality,
        "quality_percentile":     state.get("quality_percentile"),
        "cost_efficiency_ratio":  state.get("cost_efficiency_ratio"),
        "is_anomaly":             is_anomaly,
        "anomaly_reason":         state.get("anomaly_reason"),
        "performance_narrative":  state.get("performance_narrative"),
        "network_recommendation": recommendation,
        "agents_executed":        state.get("agents_executed", []),
        "agents_skipped":         state.get("agents_skipped", []),
        "scorecard_version":      SCORECARD_VERSION,
        "generated_at":           datetime.now(UTC).isoformat(),
    }

    with get_db() as db:
        update_provider_recommendation(db, npi,
            recommendation=recommendation,
            scorecard_version=SCORECARD_VERSION,
        )

    executed = state.get("agents_executed", []) + ["reporter"]
    logger.info(f"[Reporter] NPI={npi} recommendation={recommendation}")

    return {
        **state,
        "network_recommendation": recommendation,
        "scorecard":              scorecard,
        "scorecard_version":      SCORECARD_VERSION,
        "pipeline_complete":      True,
        "agents_executed":        executed,
    }
