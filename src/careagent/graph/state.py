"""
careagent.graph.state
~~~~~~~~~~~~~~~~~~~~~
Shared AgentState TypedDict — the single source of truth
that all agents read from and write to.

Design principle: every field that any agent produces is defined
here BEFORE any agent is written. This prevents schema drift
between agents and makes the supervisor routing logic type-safe.
"""

from typing import Optional, TypedDict


class AgentState(TypedDict):
    """
    Shared state object passed between all agents in the pipeline.
    The supervisor reads this after each agent step to decide
    what runs next.

    Fields are grouped by which agent populates them.
    All fields are Optional — agents only fill their own section.
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    npi: str                          # Provider NPI — the pipeline input
    run_id: str                       # UUID for this pipeline execution

    # ── Supervisor control ─────────────────────────────────────────────────────
    next_agent: Optional[str]         # Which agent runs next (set by supervisor)
    agents_executed: list[str]        # Agents that have run
    agents_skipped: list[str]         # Agents skipped by supervisor
    error: Optional[str]              # Set if any agent fails

    # ── Data Cleaner Agent output ──────────────────────────────────────────────
    data_quality_score: Optional[float]   # 0.0-1.0. < 0.85 triggers cleaning
    fields_imputed: Optional[int]         # How many fields were imputed
    cleaning_notes: Optional[str]         # What was fixed or flagged
    cleaning_complete: Optional[bool]     # True when cleaner is done

    # ── Statistical Agent output ───────────────────────────────────────────────
    quality_score: Optional[float]        # Composite score 0-100
    quality_percentile: Optional[float]   # National percentile 0-100
    cost_efficiency_ratio: Optional[float] # submitted_charge / medicare_payment
    volume_percentile: Optional[float]    # Volume vs specialty peers
    specialty_benchmarks: Optional[dict]  # National avg metrics for specialty
    scoring_complete: Optional[bool]

    # ── Anomaly Detection Agent output ─────────────────────────────────────────
    is_anomaly: Optional[bool]            # True if Isolation Forest flagged
    anomaly_score: Optional[float]        # Raw IF score (more negative = worse)
    anomaly_reason: Optional[str]         # Plain-English explanation
    anomaly_complete: Optional[bool]

    # ── Summarizer Agent output ────────────────────────────────────────────────
    performance_narrative: Optional[str]  # GPT-generated summary
    narrative_faithfulness: Optional[float]  # RAGAS score
    narrative_relevancy: Optional[float]     # RAGAS score
    summarizer_complete: Optional[bool]

    # ── Reporter Agent output ──────────────────────────────────────────────────
    network_recommendation: Optional[str] # include | review | exclude
    scorecard: Optional[dict]             # Final structured scorecard JSON
    scorecard_version: Optional[str]
    pipeline_complete: Optional[bool]


def initial_state(npi: str, run_id: str) -> AgentState:
    """
    Create a fresh AgentState for a new pipeline run.
    All optional fields start as None — agents fill them in.
    """
    return AgentState(
        npi=npi,
        run_id=run_id,
        next_agent="supervisor",
        agents_executed=[],
        agents_skipped=[],
        error=None,
        data_quality_score=None,
        fields_imputed=None,
        cleaning_notes=None,
        cleaning_complete=None,
        quality_score=None,
        quality_percentile=None,
        cost_efficiency_ratio=None,
        volume_percentile=None,
        specialty_benchmarks=None,
        scoring_complete=None,
        is_anomaly=None,
        anomaly_score=None,
        anomaly_reason=None,
        anomaly_complete=None,
        performance_narrative=None,
        narrative_faithfulness=None,
        narrative_relevancy=None,
        summarizer_complete=None,
        network_recommendation=None,
        scorecard=None,
        scorecard_version=None,
        pipeline_complete=None,
    )
