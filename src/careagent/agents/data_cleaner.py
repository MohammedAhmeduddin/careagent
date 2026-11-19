"""
careagent.agents.data_cleaner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Data Validator agent (graph node "data_cleaner").

Checks that the fields the peer comparison depends on are present and
valid. It never imputes: a value filled in with a specialty median would
feed straight into an outlier comparison and make the provider look
average. Missing data is reported, and such providers are scored as
"insufficient_data" at load time.
"""

from loguru import logger

from careagent.db.queries import get_provider_by_npi, update_provider_validation
from careagent.db.session import get_db
from careagent.graph.state import AgentState
from careagent.scoring.peer import REQUIRED_FIELDS

POSITIVE_FIELDS = [f for f in REQUIRED_FIELDS if f != "provider_type"]


def validate(provider) -> tuple[float, list[str]]:
    """Return (completeness 0-1, list of missing or invalid fields)."""
    problems = []
    if not getattr(provider, "provider_type", None):
        problems.append("provider_type")
    for field in POSITIVE_FIELDS:
        value = getattr(provider, field, None)
        if value is None or value <= 0:
            problems.append(field)
    completeness = round(1 - len(problems) / len(REQUIRED_FIELDS), 3)
    return completeness, problems


def data_cleaner_agent(state: AgentState) -> AgentState:
    npi = state["npi"]
    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}

        completeness, problems = validate(provider)
        notes = (
            "All required fields present and valid"
            if not problems
            else f"Missing or invalid: {', '.join(problems)} (not imputed)"
        )
        update_provider_validation(db, npi, notes)
        provider_type, city, prov_state = provider.provider_type, provider.city, provider.state

    logger.info(f"[DataValidator] NPI={npi} completeness={completeness}")
    return {
        **state,
        "provider_type": provider_type,
        "city": city,
        "state": prov_state,
        "data_completeness": completeness,
        "missing_fields": problems,
        "cleaning_notes": notes,
        "cleaning_complete": True,
        "agents_executed": state.get("agents_executed", []) + ["data_cleaner"],
    }
