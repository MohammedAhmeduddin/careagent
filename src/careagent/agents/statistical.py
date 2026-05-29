"""
careagent.agents.statistical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Statistical Agent — computes composite quality score benchmarked
against specialty national averages from the database.

Quality Score Formula (0-100):
    - Cost efficiency component (40%):
        lower cost_ratio vs specialty avg = higher score
    - Volume component (30%):
        percentile rank of total_services within specialty
    - Payment efficiency component (30%):
        how close medicare_payment is to allowed_amount

All benchmarks are computed live from the providers table,
so scores update automatically as more data is loaded.
"""

import numpy as np
from loguru import logger
from sqlalchemy import func, select
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.models import Provider
from careagent.db.queries import (
    get_provider_by_npi,
    get_specialty_benchmarks,
    update_provider_quality_scores,
)


def _compute_percentile_rank(db, npi: str, field: str, provider_type: str) -> float:
    """
    Compute what percentile this provider falls in for a given field
    within their specialty. Returns 0.0-100.0.
    """
    col = getattr(Provider, field)
    provider_val = db.scalar(
        select(col).where(Provider.npi == npi)
    )
    if provider_val is None:
        return 50.0  # Default to median if missing

    below_count = db.scalar(
        select(func.count(Provider.npi))
        .where(Provider.provider_type == provider_type)
        .where(col < provider_val)
        .where(col.is_not(None))
    ) or 0

    total_count = db.scalar(
        select(func.count(Provider.npi))
        .where(Provider.provider_type == provider_type)
        .where(col.is_not(None))
    ) or 1

    return round((below_count / total_count) * 100, 2)


def _compute_quality_score(
    provider,
    benchmarks: dict,
    volume_percentile: float,
) -> float:
    """
    Composite quality score 0-100.

    Components:
    - Cost efficiency (40%): lower submitted_charge/allowed_amount = better
    - Volume (30%): higher volume percentile = better
    - Payment efficiency (30%): medicare_payment/allowed_amount ratio
    """
    scores = []
    weights = []

    # ── Cost efficiency (40%) ──────────────────────────────────────────────────
    # Compare provider's cost ratio to specialty average
    if provider.avg_submitted_charge and provider.avg_allowed_amount:
        prov_ratio = provider.avg_submitted_charge / max(provider.avg_allowed_amount, 1)
        avg_charge = benchmarks.get("avg_charge") or provider.avg_submitted_charge
        avg_allowed = benchmarks.get("avg_payment") or provider.avg_allowed_amount
        bench_ratio = avg_charge / max(avg_allowed, 1)

        # Score: 100 if at benchmark, decreasing as ratio rises above benchmark
        cost_score = max(0.0, min(100.0, 100.0 - (prov_ratio - bench_ratio) * 30))
        scores.append(cost_score)
        weights.append(0.40)

    # ── Volume (30%) ───────────────────────────────────────────────────────────
    scores.append(volume_percentile)
    weights.append(0.30)

    # ── Payment efficiency (30%) ───────────────────────────────────────────────
    if provider.avg_medicare_payment and provider.avg_allowed_amount:
        pay_ratio = provider.avg_medicare_payment / max(provider.avg_allowed_amount, 1)
        # Higher ratio (closer to 1.0) = Medicare paying close to allowed = better
        pay_score = min(100.0, pay_ratio * 100.0)
        scores.append(pay_score)
        weights.append(0.30)

    if not scores:
        return 50.0

    # Normalize weights
    total_weight = sum(weights)
    normalized = [w / total_weight for w in weights]
    return round(sum(s * w for s, w in zip(scores, normalized)), 2)


def statistical_agent(state: AgentState) -> AgentState:
    """
    Scores provider against specialty national benchmarks.

    Produces:
    - quality_score: composite 0-100
    - quality_percentile: national percentile within specialty
    - cost_efficiency_ratio: submitted_charge / medicare_payment
    - volume_percentile: services volume rank within specialty
    """
    npi = state["npi"]
    logger.info(f"[Statistical] Processing NPI={npi}")

    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}

        specialty = provider.provider_type

        # ── Benchmarks ─────────────────────────────────────────────────────────
        benchmarks = get_specialty_benchmarks(db, specialty)
        if not benchmarks:
            logger.warning(f"[Statistical] No benchmarks for {specialty} — using defaults")
            benchmarks = {
                "avg_payment": 150.0, "avg_charge": 350.0,
                "avg_services": 500.0, "avg_beneficiaries": 200.0,
                "provider_count": 1,
            }

        # ── Volume percentile ──────────────────────────────────────────────────
        volume_percentile = _compute_percentile_rank(
            db, npi, "total_services", specialty
        )

        # ── Quality percentile (based on payment efficiency) ───────────────────
        quality_percentile = _compute_percentile_rank(
            db, npi, "avg_medicare_payment", specialty
        )

        # ── Cost efficiency ratio ──────────────────────────────────────────────
        cost_efficiency_ratio = round(
            (provider.avg_submitted_charge or 0) /
            max(provider.avg_medicare_payment or 1, 1),
            3,
        )

        # ── Composite quality score ────────────────────────────────────────────
        quality_score = _compute_quality_score(
            provider, benchmarks, volume_percentile
        )

        # ── Write to database ──────────────────────────────────────────────────
        update_provider_quality_scores(
            db, npi,
            quality_score=quality_score,
            quality_percentile=quality_percentile,
            cost_efficiency_ratio=cost_efficiency_ratio,
            volume_percentile=volume_percentile,
        )

    executed = state.get("agents_executed", []) + ["statistical"]
    logger.info(
        f"[Statistical] NPI={npi} score={quality_score} "
        f"percentile={quality_percentile} cost_ratio={cost_efficiency_ratio}"
    )

    return {
        **state,
        "quality_score":         quality_score,
        "quality_percentile":    quality_percentile,
        "cost_efficiency_ratio": cost_efficiency_ratio,
        "volume_percentile":     volume_percentile,
        "specialty_benchmarks":  benchmarks,
        "scoring_complete":      True,
        "agents_executed":       executed,
    }
