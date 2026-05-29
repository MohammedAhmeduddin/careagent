"""
careagent.api.routes.analyze
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/analyze endpoint — runs the full CareAgent pipeline for a provider NPI.
"""

import time
import uuid
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException
from loguru import logger

from careagent.api.schemas import AnalyzeRequest, ScorecardResponse, ErrorResponse
from careagent.graph.state import initial_state
from careagent.graph.workflow import careagent_graph
from careagent.db.session import get_db
from careagent.db.queries import (
    get_provider_by_npi,
    create_agent_run,
    complete_agent_run,
    fail_agent_run,
)

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze",
    response_model=ScorecardResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def analyze_provider(request: AnalyzeRequest) -> ScorecardResponse:
    """
    Run the full CareAgent pipeline for a provider NPI.

    Pipeline steps (supervisor decides which run):
    1. Data Cleaner — validates and imputes missing fields
    2. Statistical Agent — quality score vs national benchmarks
    3. Anomaly Agent — Isolation Forest outlier detection
    4. Summarizer — GPT-4o-mini performance narrative
    5. Reporter — final scorecard + network recommendation

    Returns the complete scorecard in under 10 seconds (warm instance).
    """
    npi    = request.npi
    run_id = str(uuid.uuid4())
    start  = time.perf_counter()

    logger.info(f"[API] /analyze called — NPI={npi} run_id={run_id}")

    # ── Verify provider exists ─────────────────────────────────────────────────
    with get_db() as db:
        provider = get_provider_by_npi(db, npi)
        if not provider:
            raise HTTPException(
                status_code=404,
                detail=f"Provider NPI {npi} not found in database",
            )
        create_agent_run(db, run_id=run_id, npi=npi)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    state = initial_state(npi=npi, run_id=run_id)

    try:
        final_state = careagent_graph.invoke(state)
    except Exception as e:
        logger.error(f"[API] Pipeline failed for NPI={npi}: {e}")
        with get_db() as db:
            fail_agent_run(db, run_id=run_id, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    duration = round(time.perf_counter() - start, 3)

    # ── Check for pipeline-level errors ───────────────────────────────────────
    if final_state.get("error"):
        with get_db() as db:
            fail_agent_run(db, run_id=run_id, error_message=final_state["error"])
        raise HTTPException(status_code=500, detail=final_state["error"])

    # ── Log the completed run ──────────────────────────────────────────────────
    with get_db() as db:
        complete_agent_run(
            db,
            run_id=run_id,
            agents_executed=final_state.get("agents_executed", []),
            agents_skipped=final_state.get("agents_skipped", []),
            duration_seconds=duration,
            tokens_used=0,
            llm_calls=1 if final_state.get("performance_narrative") else 0,
            recommendation=final_state.get("network_recommendation", "review"),
        )
        provider = get_provider_by_npi(db, npi)

    logger.info(
        f"[API] Pipeline complete — NPI={npi} "
        f"duration={duration}s "
        f"recommendation={final_state.get('network_recommendation')}"
    )

    # ── Build response ─────────────────────────────────────────────────────────
    return ScorecardResponse(
        npi=npi,
        run_id=run_id,
        provider_name=f"{provider.last_name_or_org}, {provider.first_name or ''}".strip(", "),
        provider_type=provider.provider_type,
        state=provider.state,
        quality_score=final_state.get("quality_score"),
        quality_percentile=final_state.get("quality_percentile"),
        cost_efficiency_ratio=final_state.get("cost_efficiency_ratio"),
        volume_percentile=final_state.get("volume_percentile"),
        is_anomaly=final_state.get("is_anomaly"),
        anomaly_score=final_state.get("anomaly_score"),
        anomaly_reason=final_state.get("anomaly_reason"),
        data_quality_score=final_state.get("data_quality_score"),
        fields_imputed=final_state.get("fields_imputed"),
        performance_narrative=final_state.get("performance_narrative"),
        narrative_faithfulness=final_state.get("narrative_faithfulness"),
        narrative_relevancy=final_state.get("narrative_relevancy"),
        network_recommendation=final_state.get("network_recommendation"),
        scorecard_version=final_state.get("scorecard_version"),
        agents_executed=final_state.get("agents_executed", []),
        agents_skipped=final_state.get("agents_skipped", []),
        pipeline_duration_seconds=duration,
        generated_at=datetime.now(UTC),
    )
