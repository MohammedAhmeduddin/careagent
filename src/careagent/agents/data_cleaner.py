"""
careagent.agents.data_cleaner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data Cleaner Agent — validates and imputes missing provider fields
using specialty-level medians from the database.

Real imputation strategy:
- Missing numeric fields → imputed with specialty median
- Missing categorical fields → flagged, not imputed
- Data quality score = fraction of key fields present BEFORE imputation
"""

from loguru import logger
from sqlalchemy import func, select
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.models import Provider
from careagent.db.queries import get_provider_by_npi, update_provider_cleaning

# Fields that must be present for agents to work correctly
KEY_NUMERIC_FIELDS = [
    "avg_medicare_payment",
    "avg_submitted_charge",
    "avg_allowed_amount",
    "total_services",
    "total_unique_beneficiaries",
]
KEY_CATEGORICAL_FIELDS = ["state", "provider_type"]
ALL_KEY_FIELDS = KEY_NUMERIC_FIELDS + KEY_CATEGORICAL_FIELDS


def _get_specialty_medians(db, provider_type: str) -> dict:
    """
    Compute median values for numeric fields within a specialty.
    Used to impute missing values for a provider.
    """
    medians = {}
    for field in KEY_NUMERIC_FIELDS:
        col = getattr(Provider, field)
        # PostgreSQL percentile_cont for true median
        result = db.execute(
            select(func.percentile_cont(0.5).within_group(col))
            .where(Provider.provider_type == provider_type)
            .where(col.is_not(None))
        ).scalar()
        medians[field] = result
    return medians


def data_cleaner_agent(state: AgentState) -> AgentState:
    """
    Validates provider data quality and imputes missing fields.

    Steps:
    1. Assess data quality score (before imputation)
    2. Fetch specialty medians for imputation
    3. Impute missing numeric fields
    4. Flag missing categorical fields
    5. Write results to state and database
    """
    npi = state["npi"]
    logger.info(f"[DataCleaner] Processing NPI={npi}")

    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}

        # ── Step 1: Assess quality BEFORE imputation ───────────────────────────
        missing_fields = [
            f for f in ALL_KEY_FIELDS
            if getattr(provider, f, None) is None
        ]
        quality_score = round(
            1.0 - len(missing_fields) / len(ALL_KEY_FIELDS), 3
        )

        if not missing_fields:
            notes = "All key fields present — no imputation needed"
            update_provider_cleaning(
                db, npi,
                data_quality_score=quality_score,
                fields_imputed=0,
                cleaning_notes=notes,
            )
            executed = state.get("agents_executed", []) + ["data_cleaner"]
            logger.info(f"[DataCleaner] NPI={npi} quality=1.0 no imputation needed")
            return {
                **state,
                "data_quality_score": quality_score,
                "fields_imputed":     0,
                "cleaning_notes":     notes,
                "cleaning_complete":  True,
                "agents_executed":    executed,
            }

        # ── Step 2: Get specialty medians ──────────────────────────────────────
        medians = _get_specialty_medians(db, provider.provider_type)

        # ── Step 3: Impute missing numeric fields ──────────────────────────────
        imputed = []
        impute_values = {}

        for field in KEY_NUMERIC_FIELDS:
            if getattr(provider, field, None) is None:
                median_val = medians.get(field)
                if median_val is not None:
                    impute_values[field] = float(median_val)
                    imputed.append(f"{field}={median_val:.2f}(median)")
                    setattr(provider, field, float(median_val))

        # Apply imputed values to database
        if impute_values:
            for field, val in impute_values.items():
                setattr(provider, field, val)
            db.flush()

        # ── Step 4: Flag missing categorical fields ────────────────────────────
        missing_cats = [
            f for f in KEY_CATEGORICAL_FIELDS
            if getattr(provider, f, None) is None
        ]
        if missing_cats:
            imputed.append(f"missing categoricals: {missing_cats}")

        # ── Step 5: Write results ──────────────────────────────────────────────
        notes = (
            f"Imputed {len(impute_values)} fields: {'; '.join(imputed)}"
            if imputed else "No imputation needed"
        )

        update_provider_cleaning(
            db, npi,
            data_quality_score=quality_score,
            fields_imputed=len(impute_values),
            cleaning_notes=notes,
        )

    executed = state.get("agents_executed", []) + ["data_cleaner"]
    logger.info(
        f"[DataCleaner] NPI={npi} quality={quality_score:.3f} "
        f"imputed={len(impute_values)}"
    )

    return {
        **state,
        "data_quality_score": quality_score,
        "fields_imputed":     len(impute_values),
        "cleaning_notes":     notes,
        "cleaning_complete":  True,
        "agents_executed":    executed,
    }
