"""
tests/unit/test_agents_coverage.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Additional tests to cover missing lines in agents and db/session.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from careagent.db.models import Base, Provider
from careagent.db.session import check_connection, create_tables
from careagent.graph.state import initial_state
from careagent.agents.summarizer import _build_prompt, _template_narrative
from careagent.agents.reporter import reporter_agent
from careagent.agents.data_cleaner import data_cleaner_agent


# ── Fixtures ───────────────────────────────────────────────────────────────────

@dataclass
class FakeProvider:
    npi: str = "1234567890"
    last_name_or_org: str = "Smith"
    first_name: Optional[str] = "John"
    provider_type: str = "Cardiology"
    state: str = "CA"
    avg_submitted_charge: Optional[float] = 450.0
    avg_medicare_payment: Optional[float] = 200.0
    avg_allowed_amount: Optional[float] = 220.0
    total_services: Optional[float] = 800.0
    total_unique_beneficiaries: Optional[float] = 400.0


# ── Summarizer coverage ────────────────────────────────────────────────────────

class TestSummarizerPrompt:

    def test_build_prompt_contains_npi_data(self):
        state = initial_state("1234567890", "run-001")
        state = {**state, "quality_score": 78.5, "quality_percentile": 65.0,
                 "cost_efficiency_ratio": 2.1, "volume_percentile": 70.0,
                 "is_anomaly": False}
        prompt = _build_prompt(FakeProvider(), state)
        assert "Smith" in prompt
        assert "Cardiology" in prompt
        assert "78.5" in prompt

    def test_build_prompt_includes_anomaly_flag(self):
        state = initial_state("1234567890", "run-001")
        state = {**state, "quality_score": 35.0, "quality_percentile": 20.0,
                 "cost_efficiency_ratio": 4.5, "volume_percentile": 30.0,
                 "is_anomaly": True, "anomaly_reason": "High cost outlier"}
        prompt = _build_prompt(FakeProvider(), state)
        assert "ANOMALY FLAG: Yes" in prompt
        assert "High cost outlier" in prompt

    def test_build_prompt_no_anomaly(self):
        state = initial_state("1234567890", "run-001")
        state = {**state, "quality_score": 80.0, "quality_percentile": 70.0,
                 "cost_efficiency_ratio": 2.0, "volume_percentile": 60.0,
                 "is_anomaly": False}
        prompt = _build_prompt(FakeProvider(), state)
        assert "ANOMALY FLAG: None" in prompt

    def test_template_narrative_above_average(self):
        state = initial_state("1234567890", "run-001")
        state = {**state, "quality_score": 80.0, "quality_percentile": 75.0,
                 "is_anomaly": False}
        narrative = _template_narrative(FakeProvider(), state)
        assert "above national average" in narrative

    def test_template_narrative_below_average(self):
        state = initial_state("1234567890", "run-001")
        state = {**state, "quality_score": 40.0, "quality_percentile": 25.0,
                 "is_anomaly": False}
        narrative = _template_narrative(FakeProvider(), state)
        assert "below national average" in narrative

    def test_template_narrative_near_average(self):
        state = initial_state("1234567890", "run-001")
        state = {**state, "quality_score": 60.0, "quality_percentile": 50.0,
                 "is_anomaly": False}
        narrative = _template_narrative(FakeProvider(), state)
        assert "near national average" in narrative


# ── Reporter coverage ──────────────────────────────────────────────────────────

class TestReporterRecommendations:

    def _run_reporter(self, quality_score, is_anomaly):
        state = initial_state("1234567890", "run-001")
        state = {
            **state,
            "quality_score":         quality_score,
            "quality_percentile":    50.0,
            "cost_efficiency_ratio": 2.0,
            "volume_percentile":     50.0,
            "is_anomaly":            is_anomaly,
            "anomaly_score":         -0.1,
            "anomaly_reason":        "test",
            "performance_narrative": "Test narrative",
            "cleaning_complete":     True,
            "scoring_complete":      True,
            "anomaly_complete":      True,
            "summarizer_complete":   True,
            "agents_executed":       ["data_cleaner", "statistical", "anomaly", "summarizer"],
            "agents_skipped":        [],
        }
        with patch("careagent.agents.reporter.get_db") as mock_db, \
             patch("careagent.agents.reporter.get_provider_by_npi", return_value=FakeProvider()), \
             patch("careagent.agents.reporter.update_provider_recommendation"):
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            return reporter_agent(state)

    def test_high_quality_no_anomaly_returns_include(self):
        result = self._run_reporter(quality_score=80.0, is_anomaly=False)
        assert result["network_recommendation"] == "include"

    def test_low_quality_returns_review(self):
        result = self._run_reporter(quality_score=50.0, is_anomaly=False)
        assert result["network_recommendation"] == "review"

    def test_anomaly_returns_review(self):
        result = self._run_reporter(quality_score=80.0, is_anomaly=True)
        assert result["network_recommendation"] == "review"

    def test_reporter_sets_pipeline_complete(self):
        result = self._run_reporter(quality_score=80.0, is_anomaly=False)
        assert result["pipeline_complete"] is True

    def test_reporter_adds_itself_to_executed(self):
        result = self._run_reporter(quality_score=80.0, is_anomaly=False)
        assert "reporter" in result["agents_executed"]


# ── Data cleaner coverage ──────────────────────────────────────────────────────

class TestDataCleanerAgent:

    def _make_db_provider(self, missing_field=None):
        p = MagicMock()
        p.npi            = "1234567890"
        p.provider_type  = "Cardiology"
        p.avg_medicare_payment       = None if missing_field == "avg_medicare_payment" else 150.0
        p.avg_submitted_charge       = None if missing_field == "avg_submitted_charge" else 300.0
        p.avg_allowed_amount         = None if missing_field == "avg_allowed_amount"    else 160.0
        p.total_services             = None if missing_field == "total_services"        else 500.0
        p.total_unique_beneficiaries = None if missing_field == "total_unique_beneficiaries" else 200.0
        p.state                      = "CA"
        return p

    def test_cleaner_perfect_data_returns_quality_1(self):
        state = initial_state("1234567890", "run-001")
        with patch("careagent.agents.data_cleaner.get_db") as mock_db, \
             patch("careagent.agents.data_cleaner.get_provider_by_npi",
                   return_value=self._make_db_provider()), \
             patch("careagent.agents.data_cleaner.update_provider_cleaning"):
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            result = data_cleaner_agent(state)
        assert result["data_quality_score"] == 1.0
        assert result["fields_imputed"]     == 0
        assert result["cleaning_complete"]  is True

    def test_cleaner_missing_provider_returns_error(self):
        state = initial_state("9999999999", "run-001")
        with patch("careagent.agents.data_cleaner.get_db") as mock_db, \
             patch("careagent.agents.data_cleaner.get_provider_by_npi", return_value=None):
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            result = data_cleaner_agent(state)
        assert result["error"] is not None

    def test_cleaner_imputes_missing_field(self):
        state    = initial_state("1234567890", "run-001")
        provider = self._make_db_provider(missing_field="avg_medicare_payment")
        with patch("careagent.agents.data_cleaner.get_db") as mock_db, \
             patch("careagent.agents.data_cleaner.get_provider_by_npi", return_value=provider), \
             patch("careagent.agents.data_cleaner._get_specialty_medians",
                   return_value={"avg_medicare_payment": 145.0, "avg_submitted_charge": 290.0,
                                 "avg_allowed_amount": 155.0, "total_services": 480.0,
                                 "total_unique_beneficiaries": 190.0}), \
             patch("careagent.agents.data_cleaner.update_provider_cleaning"):
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            result = data_cleaner_agent(state)
        assert result["fields_imputed"]    >= 1
        assert result["cleaning_complete"] is True
        assert result["data_quality_score"] < 1.0


# ── DB session coverage ────────────────────────────────────────────────────────

class TestDbSession:

    def test_check_connection_returns_bool(self):
        result = check_connection()
        assert isinstance(result, bool)

    def test_create_tables_runs_without_error(self):
        # Uses real engine — just verify no exception raised
        try:
            create_tables()
        except Exception as e:
            pytest.fail(f"create_tables raised: {e}")

    def test_check_connection_false_on_bad_url(self):
        with patch("careagent.db.session.engine") as mock_engine:
            mock_engine.connect.side_effect = Exception("connection refused")
            result = check_connection()
            assert result is False
