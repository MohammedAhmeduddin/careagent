"""
careagent.agents.data_cleaner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data Cleaner Agent — validates and imputes missing provider fields.
Week 2: stub with correct interface. Full logic in Week 3.
"""

from loguru import logger
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.queries import get_provider_by_npi, update_provider_cleaning

KEY_FIELDS = [
    "avg_medicare_payment", "avg_submitted_charge",
    "total_services", "total_unique_beneficiaries",
    "state", "provider_type",
]


def data_cleaner_agent(state: AgentState) -> AgentState:
    """
    Assesses data quality and imputes missing fields.
    Writes results to both AgentState and the providers table.
    """
    npi = state["npi"]
    logger.info(f"[DataCleaner] Processing NPI={npi}")

    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found in database"}

        # Count missing key fields
        missing = sum(1 for f in KEY_FIELDS if getattr(provider, f) is None)
        total   = len(KEY_FIELDS)
        quality = round(1.0 - (missing / total), 3)
        notes   = f"{missing}/{total} key fields missing" if missing else "All key fields present"

        update_provider_cleaning(
            db, npi,
            data_quality_score=quality,
            fields_imputed=missing,
            cleaning_notes=notes,
        )

    executed = state.get("agents_executed", []) + ["data_cleaner"]
    logger.info(f"[DataCleaner] NPI={npi} quality={quality:.3f} imputed={missing}")

    return {
        **state,
        "data_quality_score":  quality,
        "fields_imputed":      missing,
        "cleaning_notes":      notes,
        "cleaning_complete":   True,
        "agents_executed":     executed,
    }
