"""
careagent.agents.statistical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Statistical Agent — computes quality scores vs national benchmarks.
Week 2: stub. Full scoring logic in Week 3.
"""

from loguru import logger
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.queries import (
    get_provider_by_npi,
    get_specialty_benchmarks,
    update_provider_quality_scores,
)
import numpy as np


def statistical_agent(state: AgentState) -> AgentState:
    """Scores provider against specialty national benchmarks."""
    npi = state["npi"]
    logger.info(f"[Statistical] Processing NPI={npi}")

    with get_db() as db:
        provider  = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}

        benchmarks = get_specialty_benchmarks(db, provider.provider_type)
        if not benchmarks:
            benchmarks = {"avg_payment": 150.0, "avg_charge": 350.0,
                         "avg_services": 500.0, "provider_count": 1}

        # Cost efficiency: lower submitted charge relative to payment = better
        cost_ratio = round(
            (provider.avg_submitted_charge or 0) /
            max(provider.avg_medicare_payment or 1, 1), 3
        )

        # Quality score: simple composite (stub — full formula in Week 3)
        national_avg_pay = benchmarks.get("avg_payment") or 150.0
        pay_ratio  = (provider.avg_medicare_payment or 0) / max(national_avg_pay, 1)
        quality    = round(min(max(pay_ratio * 70, 0), 100), 2)
        percentile = round(min(max(pay_ratio * 60, 0), 100), 2)
        vol_pct    = 50.0  # placeholder

        update_provider_quality_scores(
            db, npi,
            quality_score=quality,
            quality_percentile=percentile,
            cost_efficiency_ratio=cost_ratio,
            volume_percentile=vol_pct,
        )

    executed = state.get("agents_executed", []) + ["statistical"]
    logger.info(f"[Statistical] NPI={npi} score={quality} percentile={percentile}")

    return {
        **state,
        "quality_score":        quality,
        "quality_percentile":   percentile,
        "cost_efficiency_ratio": cost_ratio,
        "volume_percentile":    vol_pct,
        "specialty_benchmarks": benchmarks,
        "scoring_complete":     True,
        "agents_executed":      executed,
    }
