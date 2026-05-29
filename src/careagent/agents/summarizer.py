"""
careagent.agents.summarizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Summarizer Agent — generates provider performance narrative via GPT.
Week 2: stub with correct interface. Real GPT call in Week 4.
"""

from loguru import logger
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.queries import get_provider_by_npi, update_provider_narrative


def summarizer_agent(state: AgentState) -> AgentState:
    """Generates a plain-English provider performance narrative."""
    npi = state["npi"]
    logger.info(f"[Summarizer] Processing NPI={npi}")

    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}

        # Stub narrative — Week 4 replaces with real GPT call
        anomaly_note = " Flagged for cost review." if state.get("is_anomaly") else ""
        narrative = (
            f"{provider.last_name_or_org} is a {provider.provider_type} provider "
            f"in {provider.state} with a quality score of "
            f"{state.get('quality_score', 0):.1f} "
            f"(national percentile: {state.get('quality_percentile', 0):.1f})."
            f"{anomaly_note}"
        )

        update_provider_narrative(db, npi,
            narrative=narrative,
            faithfulness=None,
            relevancy=None,
        )

    executed = state.get("agents_executed", []) + ["summarizer"]
    logger.info(f"[Summarizer] NPI={npi} narrative generated")

    return {
        **state,
        "performance_narrative":  narrative,
        "narrative_faithfulness": None,
        "narrative_relevancy":    None,
        "summarizer_complete":    True,
        "agents_executed":        executed,
    }
