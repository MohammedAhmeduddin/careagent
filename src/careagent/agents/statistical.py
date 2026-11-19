"""
careagent.agents.statistical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Statistical agent: the provider's peer comparison.

The comparison itself (per-specialty risk adjustment, robust z-scores,
percentiles) is computed once for all providers at load time by
careagent.scoring.peer, because a percentile only means something
against the whole peer group. This agent reads the stored result and the
specialty benchmarks and puts them into the pipeline state.
"""

from loguru import logger

from careagent.db.queries import get_peer_group, get_provider_by_npi
from careagent.db.session import get_db
from careagent.graph.state import AgentState
from careagent.scoring.peer import METRICS, RAW_METRIC


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


def statistical_agent(state: AgentState) -> AgentState:
    npi = state["npi"]
    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}
        group = get_peer_group(db, provider.provider_type)

        metrics = {
            "beneficiaries": _round(provider.total_beneficiaries, 0),
            "services": _round(provider.total_services, 0),
            "distinct_hcpcs_codes": _round(provider.total_hcpcs_codes, 0),
            "avg_patient_risk_score": _round(provider.bene_avg_risk_score),
            **{RAW_METRIC[m.key]: _round(getattr(provider, RAW_METRIC[m.key])) for m in METRICS},
        }
        comparison = {
            m.key: {
                "label": m.label,
                "ratio_to_peer_median": _round(getattr(provider, f"{m.key}_ratio_to_peer_median")),
                "percentile": _round(getattr(provider, f"{m.key}_percentile"), 1),
                "modified_z": _round(getattr(provider, f"{m.key}_z"), 2),
            }
            for m in METRICS
        }
        benchmarks = None
        if group is not None:
            benchmarks = {
                "provider_type": group.provider_type,
                "peer_count": group.peer_count,
                "median_cost_per_beneficiary": _round(group.median_cost_per_beneficiary),
                "median_services_per_beneficiary": _round(group.median_services_per_beneficiary),
                "median_charge_to_allowed_ratio": _round(group.median_charge_to_allowed_ratio),
                "median_risk_score": _round(group.median_risk_score),
            }
        status = provider.billing_pattern_status
        peer_size = provider.peer_group_size

    logger.info(f"[Statistical] NPI={npi} status={status} peers={peer_size}")
    return {
        **state,
        "billing_pattern_status": status,
        "peer_group_size": peer_size,
        "provider_metrics": metrics,
        "peer_benchmarks": benchmarks,
        "peer_comparison": comparison,
        "scoring_complete": True,
        "agents_executed": state.get("agents_executed", []) + ["statistical"],
    }
