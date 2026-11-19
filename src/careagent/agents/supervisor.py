"""
careagent.agents.supervisor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Supervisor: reads AgentState after every step and decides what runs next.

Routing depends on the data, not a fixed sequence:
- A provider whose specialty has too few peers, or whose data is
  incomplete, has nothing to compare, so the Anomaly agent is skipped
  and the scorecard says why.
- Any agent error stops the run.
"""

from loguru import logger

from careagent.graph.state import AgentState
from careagent.scoring.peer import STATUS_INSUFFICIENT_DATA, STATUS_INSUFFICIENT_PEERS

# Agent name constants, used throughout the codebase
DATA_CLEANER  = "data_cleaner"
STATISTICAL   = "statistical"
ANOMALY       = "anomaly"
SUMMARIZER    = "summarizer"
REPORTER      = "reporter"
END           = "END"

NOT_COMPARABLE = {STATUS_INSUFFICIENT_PEERS, STATUS_INSUFFICIENT_DATA}


def supervisor(state: AgentState) -> AgentState:
    """Set ``next_agent``. Called by LangGraph after every agent step."""
    npi = state["npi"]

    if state.get("error"):
        logger.warning(f"[{npi}] Pipeline error, stopping: {state['error']}")
        return {**state, "next_agent": END}
    if state.get("pipeline_complete"):
        return {**state, "next_agent": END}

    if not state.get("cleaning_complete"):
        return {**state, "next_agent": DATA_CLEANER}

    if not state.get("scoring_complete"):
        return {**state, "next_agent": STATISTICAL}

    if not state.get("anomaly_complete"):
        status = state.get("billing_pattern_status")
        if status in NOT_COMPARABLE:
            logger.info(f"[{npi}] Skipping Anomaly ({status})")
            return {
                **state,
                "agents_skipped": state.get("agents_skipped", []) + [ANOMALY],
                "anomaly_complete": True,
                "is_outlier": None,
                "outlier_reasons": [],
                "next_agent": SUMMARIZER,
            }
        return {**state, "next_agent": ANOMALY}

    if not state.get("summarizer_complete"):
        return {**state, "next_agent": SUMMARIZER}

    return {**state, "next_agent": REPORTER}


def route_after_supervisor(state: AgentState) -> str:
    """LangGraph conditional edge: name of the next node."""
    return state.get("next_agent", END)
