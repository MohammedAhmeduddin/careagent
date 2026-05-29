"""
careagent.agents.supervisor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Supervisor agent — reads AgentState and decides which agent runs next.

This is the core of the multi-agent architecture. The supervisor
uses actual state values to make routing decisions, not a fixed
waterfall sequence. This means:

- A provider with clean data skips the DataCleaner
- A provider already scored this quarter skips StatisticalAgent
- A non-anomalous provider still gets a narrative (just shorter)

Interview answer to "why agents not a pipeline":
    The supervisor reads quality scores, anomaly flags, and data
    quality metrics to decide what work is actually needed.
    Two providers entering the same pipeline can take completely
    different paths through the agents.
"""

from loguru import logger
from careagent.graph.state import AgentState
from careagent.config import get_settings

settings = get_settings()

# Agent name constants — used throughout the codebase
DATA_CLEANER  = "data_cleaner"
STATISTICAL   = "statistical"
ANOMALY       = "anomaly"
SUMMARIZER    = "summarizer"
REPORTER      = "reporter"
END           = "END"


def supervisor(state: AgentState) -> AgentState:
    """
    Supervisor routing function.
    Called by LangGraph after every agent step.
    Returns updated state with next_agent set.

    Routing logic (in order):
    1. If any agent errored → END
    2. If pipeline already complete → END
    3. If data not cleaned → DataCleaner
    4. If not scored yet → StatisticalAgent
    5. If anomaly check not done → AnomalyAgent
    6. If no narrative yet → Summarizer
    7. If no scorecard yet → Reporter
    8. All done → END
    """
    npi = state["npi"]

    # ── Hard stops ─────────────────────────────────────────────────────────────
    if state.get("error"):
        logger.warning(f"[{npi}] Pipeline error — stopping: {state['error']}")
        return {**state, "next_agent": END}

    if state.get("pipeline_complete"):
        logger.info(f"[{npi}] Pipeline complete")
        return {**state, "next_agent": END}

    # ── Step 1: Data cleaning ──────────────────────────────────────────────────
    # Skip if data quality is already high enough
    if state.get("cleaning_complete") is None:
        data_quality = state.get("data_quality_score")

        if data_quality is None:
            # Haven't assessed quality yet — send to cleaner
            logger.info(f"[{npi}] → DataCleaner (quality unknown)")
            return {**state, "next_agent": DATA_CLEANER}

        if data_quality < (1.0 - settings.data_quality_threshold):
            # Quality below threshold — needs cleaning
            logger.info(f"[{npi}] → DataCleaner (quality={data_quality:.2f} below threshold)")
            return {**state, "next_agent": DATA_CLEANER}

        # Quality is fine — skip cleaner
        skipped = state.get("agents_skipped", []) + [DATA_CLEANER]
        logger.info(f"[{npi}] Skipping DataCleaner (quality={data_quality:.2f} ✓)")
        return {**state, "agents_skipped": skipped, "cleaning_complete": True, "next_agent": STATISTICAL}

    # ── Step 2: Statistical scoring ────────────────────────────────────────────
    if not state.get("scoring_complete"):
        logger.info(f"[{npi}] → StatisticalAgent")
        return {**state, "next_agent": STATISTICAL}

    # ── Step 3: Anomaly detection ──────────────────────────────────────────────
    if not state.get("anomaly_complete"):
        logger.info(f"[{npi}] → AnomalyAgent")
        return {**state, "next_agent": ANOMALY}

    # ── Step 4: Performance narrative ─────────────────────────────────────────
    if not state.get("summarizer_complete"):
        logger.info(f"[{npi}] → Summarizer")
        return {**state, "next_agent": SUMMARIZER}

    # ── Step 5: Report assembly ────────────────────────────────────────────────
    if not state.get("pipeline_complete"):
        logger.info(f"[{npi}] → Reporter")
        return {**state, "next_agent": REPORTER}

    # ── All done ───────────────────────────────────────────────────────────────
    return {**state, "next_agent": END}


def route_after_supervisor(state: AgentState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node to execute.
    Called by LangGraph to determine graph traversal.
    """
    return state.get("next_agent", END)
