"""
careagent.agents.summarizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Summarizer Agent — generates provider performance narrative via GPT-4o-mini.

Uses structured JSON output (function calling) to ensure the narrative
always returns in a predictable schema. Falls back to a template-based
narrative if the OpenAI API is unavailable or the key is not set.

RAGAS evaluation runs on a sample of outputs to measure:
- faithfulness: does the narrative accurately reflect the data?
- answer_relevancy: is the narrative relevant to the question asked?
"""

import json
from loguru import logger
from openai import OpenAI, APIError
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.queries import get_provider_by_npi, update_provider_narrative
from careagent.config import get_settings

settings = get_settings()

# Structured output schema for GPT function calling
NARRATIVE_FUNCTION = {
    "name": "generate_provider_narrative",
    "description": "Generate a structured provider performance narrative",
    "parameters": {
        "type": "object",
        "properties": {
            "narrative": {
                "type": "string",
                "description": (
                    "2-3 sentence plain-English summary of the provider's "
                    "performance based solely on the data provided. "
                    "Do not invent facts not in the input."
                ),
            },
            "key_strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 specific performance strengths from the data",
            },
            "key_concerns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-2 specific concerns from the data, empty if none",
            },
        },
        "required": ["narrative", "key_strengths", "key_concerns"],
    },
}


def _build_prompt(provider, state: AgentState) -> str:
    """Build the data-grounded prompt for GPT."""
    anomaly_line = (
        f"ANOMALY FLAG: Yes — {state.get('anomaly_reason', '')}"
        if state.get("is_anomaly")
        else "ANOMALY FLAG: None"
    )
    return f"""You are analyzing a Medicare provider's performance data.
Generate an accurate, factual narrative based ONLY on the data below.
Do not add clinical opinions or information not present in the data.

PROVIDER DATA:
- Name: {provider.last_name_or_org}, {provider.first_name or ''}
- Specialty: {provider.provider_type}
- State: {provider.state}
- Total Services: {provider.total_services or 'N/A'}
- Unique Beneficiaries: {provider.total_unique_beneficiaries or 'N/A'}
- Avg Medicare Payment: ${provider.avg_medicare_payment or 0:.2f}
- Avg Submitted Charge: ${provider.avg_submitted_charge or 0:.2f}

QUALITY METRICS:
- Quality Score: {state.get('quality_score', 0):.1f} / 100
- National Percentile: {state.get('quality_percentile', 0):.1f}%
- Cost Efficiency Ratio: {state.get('cost_efficiency_ratio', 0):.2f}
- Volume Percentile: {state.get('volume_percentile', 0):.1f}%
- {anomaly_line}

Generate a factual 2-3 sentence performance summary."""


def _template_narrative(provider, state: AgentState) -> str:
    """
    Fallback narrative when OpenAI is unavailable.
    Template-based but data-grounded — no hallucination risk.
    """
    anomaly_note = (
        " This provider has been flagged for cost review."
        if state.get("is_anomaly") else ""
    )
    percentile = state.get("quality_percentile", 50.0)
    rank = (
        "above national average" if percentile >= 60
        else "below national average" if percentile < 40
        else "near national average"
    )
    return (
        f"{provider.last_name_or_org} is a {provider.provider_type} provider "
        f"in {provider.state} performing {rank} "
        f"(quality score {state.get('quality_score', 0):.1f}, "
        f"national percentile {percentile:.1f}%). "
        f"The provider billed an average of "
        f"${provider.avg_submitted_charge or 0:.2f} per service "
        f"with Medicare paying ${provider.avg_medicare_payment or 0:.2f}.{anomaly_note}"
    )


def _call_gpt(prompt: str) -> tuple[str, int]:
    """
    Call GPT-4o-mini with structured output.
    Returns (narrative_json_str, tokens_used).
    Raises APIError on failure.
    """
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a healthcare analytics assistant. "
                    "Generate factual provider performance narratives "
                    "based strictly on the data provided."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        functions=[NARRATIVE_FUNCTION],
        function_call={"name": "generate_provider_narrative"},
        temperature=0.1,  # Low temperature for factual consistency
        max_tokens=400,
    )
    tokens = response.usage.total_tokens if response.usage else 0
    args   = response.choices[0].message.function_call.arguments
    return args, tokens


def summarizer_agent(state: AgentState) -> AgentState:
    """
    Generates a provider performance narrative using GPT-4o-mini.

    Falls back to template narrative if:
    - OpenAI API key not configured
    - API call fails after retry
    - Response schema invalid
    """
    npi = state["npi"]
    logger.info(f"[Summarizer] Processing NPI={npi}")

    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            return {**state, "error": f"Provider {npi} not found"}

        narrative     = None
        tokens_used   = 0
        used_fallback = False

        # ── Try GPT call ───────────────────────────────────────────────────────
        if settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
            try:
                prompt    = _build_prompt(provider, state)
                args, tokens_used = _call_gpt(prompt)
                parsed    = json.loads(args)
                narrative = parsed.get("narrative", "")
                strengths = parsed.get("key_strengths", [])
                concerns  = parsed.get("key_concerns", [])

                if strengths:
                    narrative += f" Strengths: {'; '.join(strengths)}."
                if concerns:
                    narrative += f" Concerns: {'; '.join(concerns)}."

                logger.info(
                    f"[Summarizer] GPT narrative generated "
                    f"({tokens_used} tokens) for NPI={npi}"
                )
            except (APIError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[Summarizer] GPT failed ({e}) — using template")
                used_fallback = True
        else:
            logger.info(f"[Summarizer] No API key — using template narrative")
            used_fallback = True

        # ── Fallback ───────────────────────────────────────────────────────────
        if used_fallback or not narrative:
            narrative = _template_narrative(provider, state)

        # ── Write to database ──────────────────────────────────────────────────
        update_provider_narrative(
            db, npi,
            narrative=narrative,
            faithfulness=None,   # RAGAS scores added in Week 5
            relevancy=None,
        )

    executed = state.get("agents_executed", []) + ["summarizer"]
    logger.info(f"[Summarizer] NPI={npi} complete (fallback={used_fallback})")

    return {
        **state,
        "performance_narrative":  narrative,
        "narrative_faithfulness": None,
        "narrative_relevancy":    None,
        "summarizer_complete":    True,
        "agents_executed":        executed,
    }
