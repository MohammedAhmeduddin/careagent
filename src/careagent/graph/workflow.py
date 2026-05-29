"""
careagent.graph.workflow
~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph StateGraph definition — wires all agents together.
The supervisor is called after every agent step to decide what runs next.
"""

from langgraph.graph import StateGraph, END
from careagent.graph.state import AgentState
from careagent.agents.supervisor import (
    supervisor, route_after_supervisor,
    DATA_CLEANER, STATISTICAL, ANOMALY, SUMMARIZER, REPORTER,
)
from careagent.agents.data_cleaner import data_cleaner_agent
from careagent.agents.statistical  import statistical_agent
from careagent.agents.anomaly       import anomaly_agent
from careagent.agents.summarizer    import summarizer_agent
from careagent.agents.reporter      import reporter_agent


def build_graph() -> StateGraph:
    """
    Build and compile the CareAgent LangGraph StateGraph.

    Graph structure:
        supervisor → [data_cleaner | statistical | anomaly | summarizer | reporter | END]
        each agent → supervisor (always returns to supervisor after each step)

    The supervisor reads state after every step and routes dynamically.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    graph.add_node("supervisor",    supervisor)
    graph.add_node(DATA_CLEANER,    data_cleaner_agent)
    graph.add_node(STATISTICAL,     statistical_agent)
    graph.add_node(ANOMALY,         anomaly_agent)
    graph.add_node(SUMMARIZER,      summarizer_agent)
    graph.add_node(REPORTER,        reporter_agent)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Supervisor routes to any agent or END ──────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            DATA_CLEANER: DATA_CLEANER,
            STATISTICAL:  STATISTICAL,
            ANOMALY:      ANOMALY,
            SUMMARIZER:   SUMMARIZER,
            REPORTER:     REPORTER,
            "END":        END,
        },
    )

    # ── Every agent returns to supervisor after completing ─────────────────────
    for agent_node in [DATA_CLEANER, STATISTICAL, ANOMALY, SUMMARIZER, REPORTER]:
        graph.add_edge(agent_node, "supervisor")

    return graph.compile()


# Singleton — import this anywhere you need to run the pipeline
careagent_graph = build_graph()
