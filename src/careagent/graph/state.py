"""
careagent.graph.state
~~~~~~~~~~~~~~~~~~~~~
Shared AgentState TypedDict: the single contract every agent reads from
and writes to. Every field an agent produces is declared here first, so
agents cannot drift apart and the supervisor's routing stays type-safe.
"""

from typing import Optional, TypedDict


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    npi: str
    run_id: str

    # ── Supervisor control ─────────────────────────────────────────────────────
    next_agent: Optional[str]
    agents_executed: list[str]
    agents_skipped: list[str]
    error: Optional[str]

    # ── Data Validator (node "data_cleaner") ───────────────────────────────────
    provider_type: Optional[str]
    city: Optional[str]
    state: Optional[str]
    data_completeness: Optional[float]   # share of required fields present, 0-1
    missing_fields: Optional[list[str]]
    cleaning_notes: Optional[str]
    cleaning_complete: Optional[bool]

    # ── Statistical Agent: peer comparison ─────────────────────────────────────
    billing_pattern_status: Optional[str]   # typical | outlier | insufficient_peers | insufficient_data
    peer_group_size: Optional[int]
    provider_metrics: Optional[dict]        # the provider's own values
    peer_benchmarks: Optional[dict]         # specialty medians
    peer_comparison: Optional[dict]         # ratio to median, percentile, z per metric
    scoring_complete: Optional[bool]

    # ── Anomaly Agent ──────────────────────────────────────────────────────────
    is_outlier: Optional[bool]
    outlier_reasons: Optional[list[str]]
    isolation_forest_percentile: Optional[float]   # 100 = most unusual in specialty
    anomaly_complete: Optional[bool]

    # ── Summarizer Agent ───────────────────────────────────────────────────────
    performance_narrative: Optional[str]
    narrative_source: Optional[str]   # "llm" or "template"
    narrative_cached: Optional[bool]  # True if an unchanged LLM narrative was reused
    llm_tokens_used: Optional[int]
    summarizer_complete: Optional[bool]

    # ── Reporter Agent ─────────────────────────────────────────────────────────
    scorecard: Optional[dict]
    scorecard_version: Optional[str]
    pipeline_complete: Optional[bool]


def initial_state(npi: str, run_id: str) -> AgentState:
    """Fresh state for a pipeline run. Agents fill in their own sections."""
    return AgentState(
        npi=npi,
        run_id=run_id,
        next_agent="supervisor",
        agents_executed=[],
        agents_skipped=[],
        error=None,
        provider_type=None,
        city=None,
        state=None,
        data_completeness=None,
        missing_fields=None,
        cleaning_notes=None,
        cleaning_complete=None,
        billing_pattern_status=None,
        peer_group_size=None,
        provider_metrics=None,
        peer_benchmarks=None,
        peer_comparison=None,
        scoring_complete=None,
        is_outlier=None,
        outlier_reasons=None,
        isolation_forest_percentile=None,
        anomaly_complete=None,
        performance_narrative=None,
        narrative_source=None,
        narrative_cached=False,
        llm_tokens_used=0,
        summarizer_complete=None,
        scorecard=None,
        scorecard_version=None,
        pipeline_complete=None,
    )
