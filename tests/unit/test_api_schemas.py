"""
tests/unit/test_api_schemas.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Pydantic v2 API schemas.
Tests validation rules — no database or API calls needed.
"""

import pytest
from datetime import datetime, UTC
from pydantic import ValidationError
from careagent.api.schemas import AnalyzeRequest, ScorecardResponse, HealthResponse


class TestAnalyzeRequest:

    def test_valid_npi_accepted(self):
        req = AnalyzeRequest(npi="1234567890")
        assert req.npi == "1234567890"

    def test_npi_too_short_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(npi="12345")

    def test_npi_too_long_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(npi="12345678901")

    def test_npi_with_letters_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeRequest(npi="123456789A")

    def test_force_rerun_defaults_to_false(self):
        req = AnalyzeRequest(npi="1234567890")
        assert req.force_rerun is False

    def test_force_rerun_can_be_set_true(self):
        req = AnalyzeRequest(npi="1234567890", force_rerun=True)
        assert req.force_rerun is True


class TestHealthResponse:

    def test_healthy_response(self):
        r = HealthResponse(
            status="healthy",
            database="connected",
            providers_loaded=10000,
            providers_scored=5000,
            anomalies_flagged=312,
        )
        assert r.status == "healthy"
        assert r.providers_loaded == 10000
        assert r.version == "0.1.0"

    def test_degraded_response(self):
        r = HealthResponse(
            status="degraded",
            database="unreachable",
            providers_loaded=0,
            providers_scored=0,
            anomalies_flagged=0,
        )
        assert r.status == "degraded"


class TestScorecardResponse:

    def _make_scorecard(self, **overrides):
        base = dict(
            npi="1234567890",
            run_id="test-run-001",
            provider_name="Smith, John",
            provider_type="Internal Medicine",
            state="NY",
            quality_score=74.3,
            quality_percentile=62.0,
            cost_efficiency_ratio=1.93,
            volume_percentile=55.0,
            is_anomaly=False,
            anomaly_score=-0.05,
            anomaly_reason="Within normal range",
            data_quality_score=0.92,
            fields_imputed=1,
            performance_narrative="Provider performs above average.",
            narrative_faithfulness=None,
            narrative_relevancy=None,
            network_recommendation="include",
            scorecard_version="0.1.0",
            agents_executed=["data_cleaner", "statistical", "anomaly", "summarizer", "reporter"],
            agents_skipped=[],
            pipeline_duration_seconds=6.2,
            generated_at=datetime.now(UTC),
        )
        return ScorecardResponse(**{**base, **overrides})

    def test_valid_scorecard_created(self):
        s = self._make_scorecard()
        assert s.npi == "1234567890"
        assert s.quality_score == 74.3
        assert s.network_recommendation == "include"

    def test_all_optional_fields_can_be_none(self):
        s = self._make_scorecard(
            quality_score=None,
            is_anomaly=None,
            performance_narrative=None,
        )
        assert s.quality_score is None
        assert s.is_anomaly is None

    def test_agents_executed_is_list(self):
        s = self._make_scorecard()
        assert isinstance(s.agents_executed, list)
        assert "statistical" in s.agents_executed

    def test_skipped_agents_recorded(self):
        s = self._make_scorecard(agents_skipped=["data_cleaner"])
        assert "data_cleaner" in s.agents_skipped


class TestSummarizerFallback:
    """Test the template narrative fallback — no OpenAI key needed."""

    def test_template_narrative_contains_provider_name(self):
        from dataclasses import dataclass
        from typing import Optional
        from careagent.agents.summarizer import _template_narrative
        from careagent.graph.state import initial_state

        @dataclass
        class FakeProvider:
            last_name_or_org: str = "Williams"
            first_name: Optional[str] = "Sarah"
            provider_type: str = "Cardiology"
            state: str = "CA"
            avg_submitted_charge: float = 450.0
            avg_medicare_payment: float = 210.0

        state = initial_state("1234567890", "test-run")
        state = {**state, "quality_score": 78.5, "quality_percentile": 71.0, "is_anomaly": False}
        narrative = _template_narrative(FakeProvider(), state)

        assert "Williams" in narrative
        assert "Cardiology" in narrative
        assert "CA" in narrative

    def test_template_narrative_flags_anomaly(self):
        from dataclasses import dataclass
        from typing import Optional
        from careagent.agents.summarizer import _template_narrative
        from careagent.graph.state import initial_state

        @dataclass
        class FakeProvider:
            last_name_or_org: str = "Jones"
            first_name: Optional[str] = None
            provider_type: str = "Orthopedic Surgery"
            state: str = "TX"
            avg_submitted_charge: float = 2000.0
            avg_medicare_payment: float = 300.0

        state = initial_state("9876543210", "test-run")
        state = {**state, "quality_score": 35.0, "quality_percentile": 22.0, "is_anomaly": True}
        narrative = _template_narrative(FakeProvider(), state)
        assert "flagged" in narrative.lower() or "review" in narrative.lower()
