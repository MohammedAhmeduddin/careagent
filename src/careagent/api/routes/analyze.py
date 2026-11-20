"""
careagent.api.routes.analyze
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/analyze endpoint — runs the full CareAgent pipeline for a provider NPI.
"""

import time
import uuid
from datetime import datetime, UTC

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from careagent.api.ratelimit import limit_analyze
from careagent.api.schemas import AnalyzeRequest, ScorecardResponse, ErrorResponse
from careagent.graph.state import initial_state
from careagent.graph.workflow import careagent_graph
from careagent.scoring.peer import DISCLAIMER
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
    responses={404: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
               500: {"model": ErrorResponse}},
    dependencies=[Depends(limit_analyze)],
)
def analyze_provider(request: AnalyzeRequest) -> ScorecardResponse:
    """
    Run the full CareAgent pipeline for a provider NPI.

    Pipeline steps (the supervisor decides which run):
    1. Data Validator: required fields present and valid (never imputes)
    2. Statistical: risk-adjusted comparison with same-specialty peers
    3. Anomaly: outlier rule + Isolation Forest rank, with reasons
       (skipped when the specialty has too few peers or data is missing)
    4. Summarizer: LLM narrative with a number check (template fallback)
    5. Reporter: final scorecard with disclaimer
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
        # Full details go to the logs and the agent_runs table, never to the
        # caller: exception text can contain hostnames or connection details.
        logger.exception(f"[API] Pipeline failed for NPI={npi} run_id={run_id}")
        with get_db() as db:
            fail_agent_run(db, run_id=run_id, error_message=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed (run_id={run_id}). Details are in the server logs.",
        )

    duration = round(time.perf_counter() - start, 3)

    # ── Check for pipeline-level errors ───────────────────────────────────────
    if final_state.get("error"):
        logger.error(
            f"[API] Pipeline error for NPI={npi} run_id={run_id}: {final_state['error']}"
        )
        with get_db() as db:
            fail_agent_run(db, run_id=run_id, error_message=final_state["error"])
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed (run_id={run_id}). Details are in the server logs.",
        )

    # ── Log the completed run ──────────────────────────────────────────────────
    with get_db() as db:
        complete_agent_run(
            db,
            run_id=run_id,
            agents_executed=final_state.get("agents_executed", []),
            agents_skipped=final_state.get("agents_skipped", []),
            duration_seconds=duration,
            tokens_used=final_state.get("llm_tokens_used") or 0,
            llm_calls=1 if (final_state.get("narrative_source") == "llm"
                            and not final_state.get("narrative_cached")) else 0,
            final_status=final_state.get("billing_pattern_status"),
        )

    logger.info(
        f"[API] Pipeline complete — NPI={npi} "
        f"duration={duration}s "
        f"status={final_state.get('billing_pattern_status')}"
    )

    # ── Build response ─────────────────────────────────────────────────────────
    return ScorecardResponse(
        npi=npi,
        run_id=run_id,
        provider_type=final_state.get("provider_type"),
        city=final_state.get("city"),
        state=final_state.get("state"),
        billing_pattern_status=final_state.get("billing_pattern_status"),
        peer_group_size=final_state.get("peer_group_size"),
        provider_metrics=final_state.get("provider_metrics"),
        peer_benchmarks=final_state.get("peer_benchmarks"),
        peer_comparison=final_state.get("peer_comparison"),
        is_outlier=final_state.get("is_outlier"),
        outlier_reasons=final_state.get("outlier_reasons") or [],
        isolation_forest_percentile=final_state.get("isolation_forest_percentile"),
        data_completeness=final_state.get("data_completeness"),
        missing_fields=final_state.get("missing_fields") or [],
        performance_narrative=final_state.get("performance_narrative"),
        narrative_source=final_state.get("narrative_source"),
        narrative_cached=bool(final_state.get("narrative_cached")),
        disclaimer=DISCLAIMER,
        scorecard_version=final_state.get("scorecard_version"),
        agents_executed=final_state.get("agents_executed", []),
        agents_skipped=final_state.get("agents_skipped", []),
        pipeline_duration_seconds=duration,
        generated_at=datetime.now(UTC),
    )
