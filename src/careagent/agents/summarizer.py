"""
careagent.agents.summarizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Summarizer agent: a short plain-English narrative of the peer comparison.

The configured OpenAI model writes it with structured (tool) output. Before
it is accepted, every number in the narrative is checked against the facts
the model was given (``numbers_grounded``); a narrative with an unsupported
number is rejected and a deterministic template is used instead. The
template is also used when no API key is set or the call fails.
"""

import json
import re

from loguru import logger
from openai import BadRequestError, OpenAI

from careagent.config import get_settings
from careagent.db.queries import update_provider_narrative
from careagent.db.session import get_db
from careagent.graph.state import AgentState
from careagent.scoring.peer import (
    DISCLAIMER, MIN_PEERS, STATUS_INSUFFICIENT_DATA, STATUS_INSUFFICIENT_PEERS,
)

settings = get_settings()

NARRATIVE_FUNCTION = {
    "name": "generate_provider_narrative",
    "description": "Summarize a provider's billing pattern versus specialty peers",
    "parameters": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": (
                    "2-3 sentences describing how the provider's billing compares with "
                    "same-specialty peers. Use only numbers from the input, written as "
                    "given. No judgment of care quality, no suggestion of fraud."
                ),
            },
        },
        "required": ["narrative"],
    },
}
NARRATIVE_TOOL = {"type": "function", "function": NARRATIVE_FUNCTION}

_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def ordinal(value: float) -> str:
    """31 -> '31st', 12 -> '12th', 22.6 -> '23rd'."""
    n = int(round(value))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _facts(state: AgentState) -> list[str]:
    """The fact lines given to the model (also used by the template)."""
    lines = [
        f"Specialty: {state.get('provider_type') or 'unknown'}",
        f"Location: {state.get('city') or 'unknown'}, {state.get('state') or ''}".rstrip(", "),
        f"Billing pattern status: {state.get('billing_pattern_status')}",
        f"Specialty peers compared: {state.get('peer_group_size') or 0}",
    ]
    metrics = state.get("provider_metrics") or {}
    if metrics.get("beneficiaries") is not None:
        lines.append(f"Medicare beneficiaries: {metrics['beneficiaries']:.0f}")
    if metrics.get("avg_patient_risk_score") is not None:
        lines.append(f"Average patient risk score: {metrics['avg_patient_risk_score']:.2f}")
    for item in (state.get("peer_comparison") or {}).values():
        if item.get("ratio_to_peer_median") is None:
            continue
        lines.append(
            f"{item['label']}: {item['ratio_to_peer_median']:.1f}x the specialty median "
            f"({ordinal(item['percentile'])} percentile)"
        )
    for reason in state.get("outlier_reasons") or []:
        lines.append(f"Outlier reason: {reason}")
    return lines


def _build_prompt(state: AgentState) -> str:
    facts = "\n".join(f"- {line}" for line in _facts(state))
    return (
        "Summarize how this Medicare provider's billing compares with providers "
        "of the same specialty, in 2-3 sentences, using only these facts:\n"
        f"{facts}\n\n"
        "Write numbers exactly as given. Do not add numbers, do not judge quality "
        "of care and do not suggest fraud or wrongdoing."
    )


def _numbers(text: str) -> set[float]:
    out = set()
    for match in _NUMBER.findall(text):
        try:
            out.add(round(float(match.replace(",", "").rstrip(".")), 2))
        except ValueError:
            continue
    return out


def numbers_grounded(narrative: str, facts: list[str]) -> bool:
    """True if every number in the narrative appears in the facts.

    Small counting words like "2 metrics" (integers up to 3) are allowed.
    """
    allowed = _numbers(" ".join(facts))
    for value in _numbers(narrative):
        if value <= 3 and value == int(value):
            continue
        if not any(abs(value - a) <= 0.051 for a in allowed):
            return False
    return True


def _template_narrative(state: AgentState) -> str:
    specialty = state.get("provider_type") or "Unknown specialty"
    status = state.get("billing_pattern_status")
    if status == STATUS_INSUFFICIENT_PEERS:
        return (
            f"This {specialty} provider was not compared: the dataset has "
            f"{state.get('peer_group_size') or 0} providers in this specialty, "
            f"fewer than the {MIN_PEERS} needed for a reliable comparison."
        )
    if status == STATUS_INSUFFICIENT_DATA:
        missing = ", ".join(state.get("missing_fields") or []) or "required fields"
        return f"This {specialty} provider was not compared because data is missing or invalid: {missing}."

    comparison = state.get("peer_comparison") or {}
    cost = comparison.get("cost", {})
    services = comparison.get("services", {})
    text = (
        f"Compared with {state.get('peer_group_size') or 0} {specialty} providers, "
        f"risk-adjusted Medicare cost per beneficiary is {cost.get('ratio_to_peer_median', 0):.1f}x "
        f"the specialty median and services per beneficiary is "
        f"{services.get('ratio_to_peer_median', 0):.1f}x."
    )
    reasons = state.get("outlier_reasons") or []
    if reasons:
        text += " Flagged as a billing-pattern outlier: " + "; ".join(reasons) + "."
    else:
        text += " Billing is within the typical range for the specialty."
    return text


# Optional request parameters some models reject. If the API names one of
# these in a 400 error, the call is retried once more without it.
OPTIONAL_PARAMS = ("temperature", "reasoning_effort")


def _create_completion(client: OpenAI, prompt: str, drop: frozenset = frozenset()):
    kwargs = dict(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a healthcare analytics assistant. You describe how a "
                    "provider's Medicare billing compares with same-specialty peers, "
                    "using only the numbers provided. You never judge quality of care "
                    "or suggest fraud."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        tools=[NARRATIVE_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": NARRATIVE_FUNCTION["name"]},
        },
        max_completion_tokens=settings.openai_max_completion_tokens,
    )
    if "temperature" not in drop:
        kwargs["temperature"] = settings.openai_temperature  # low = factual
    # gpt-6-luna only allows function tools on Chat Completions with
    # reasoning_effort="none"; set OPENAI_REASONING_EFFORT="" to omit it.
    if settings.openai_reasoning_effort and "reasoning_effort" not in drop:
        kwargs["reasoning_effort"] = settings.openai_reasoning_effort
    return client.chat.completions.create(**kwargs)


def _rejected_param(error: BadRequestError, already_dropped: frozenset) -> str | None:
    """The optional parameter a 400 error complains about, if any."""
    named = (getattr(error, "param", None) or "").lower()
    text = str(error).lower()
    for param in OPTIONAL_PARAMS:
        if param in already_dropped:
            continue
        if named == param or param in text:
            return param
    return None


def _call_gpt(prompt: str) -> tuple[str, int]:
    """
    Call the configured OpenAI model with structured (tool) output.
    Returns (narrative_json_str, tokens_used). Raises on failure.

    If the model rejects an optional parameter (temperature or
    reasoning_effort), the call is retried without it, at most once per
    parameter. Any other error is raised.
    """
    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    drop: frozenset = frozenset()
    while True:
        try:
            response = _create_completion(client, prompt, drop)
            break
        except BadRequestError as e:
            param = _rejected_param(e, drop)
            if param is None:
                raise
            logger.info(f"[Summarizer] Model rejects {param}; retrying without it")
            drop = drop | {param}

    tokens = response.usage.total_tokens if response.usage else 0
    args = response.choices[0].message.tool_calls[0].function.arguments
    return args, tokens


def summarizer_agent(state: AgentState) -> AgentState:
    """Write the narrative. Sets narrative_source to "llm" or "template"."""
    npi = state["npi"]
    narrative, tokens_used, source = None, 0, "template"
    comparable = state.get("billing_pattern_status") not in (
        STATUS_INSUFFICIENT_PEERS, STATUS_INSUFFICIENT_DATA,
    )

    if comparable and settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
        try:
            args, tokens_used = _call_gpt(_build_prompt(state))
            candidate = (json.loads(args).get("narrative") or "").strip()
            if candidate and numbers_grounded(candidate, _facts(state)):
                narrative, source = candidate, "llm"
            else:
                logger.warning(f"[Summarizer] NPI={npi}: narrative failed the grounding check")
        except Exception as e:  # any failure -> documented template fallback
            # OpenAI error text names the problem (model, parameter, quota) and
            # never contains the API key, so it is logged for diagnosis.
            logger.warning(
                f"[Summarizer] LLM call failed for NPI={npi}: "
                f"{type(e).__name__}: {str(e)[:300]}"
            )
            tokens_used = 0

    if narrative is None:
        narrative = _template_narrative(state)
    narrative = f"{narrative} {DISCLAIMER}"

    with get_db() as db:
        update_provider_narrative(db, npi, narrative=narrative, source=source)

    logger.info(f"[Summarizer] NPI={npi} source={source}")
    return {
        **state,
        "performance_narrative": narrative,
        "narrative_source": source,
        "llm_tokens_used": tokens_used,
        "summarizer_complete": True,
        "agents_executed": state.get("agents_executed", []) + ["summarizer"],
    }
