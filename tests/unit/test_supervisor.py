"""
tests/unit/test_supervisor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for supervisor routing logic.
These prove the dynamic routing works correctly —
the key interview justification for the multi-agent architecture.
"""

import pytest
from careagent.graph.state import initial_state
from careagent.agents.supervisor import (
    supervisor,
    DATA_CLEANER, STATISTICAL, ANOMALY, SUMMARIZER, REPORTER, END,
)


def make_state(npi="1234567890", **overrides):
    """Helper — creates a base state with optional overrides."""
    state = initial_state(npi=npi, run_id="test-run-001")
    return {**state, **overrides}


class TestSupervisorRouting:

    def test_routes_to_cleaner_when_quality_unknown(self):
        state  = make_state(data_quality_score=None)
        result = supervisor(state)
        assert result["next_agent"] == DATA_CLEANER

    def test_routes_to_cleaner_when_quality_low(self):
        state  = make_state(data_quality_score=0.70)
        result = supervisor(state)
        assert result["next_agent"] == DATA_CLEANER

    def test_skips_cleaner_when_quality_high(self):
        state  = make_state(data_quality_score=0.95)
        result = supervisor(state)
        assert result["next_agent"] == STATISTICAL
        assert DATA_CLEANER in result["agents_skipped"]

    def test_routes_to_statistical_after_cleaning(self):
        state = make_state(
            data_quality_score=0.95,
            cleaning_complete=True,
        )
        result = supervisor(state)
        assert result["next_agent"] == STATISTICAL

    def test_routes_to_anomaly_after_scoring(self):
        state = make_state(
            cleaning_complete=True,
            scoring_complete=True,
            quality_score=72.5,
        )
        result = supervisor(state)
        assert result["next_agent"] == ANOMALY

    def test_routes_to_summarizer_after_anomaly(self):
        state = make_state(
            cleaning_complete=True,
            scoring_complete=True,
            anomaly_complete=True,
            quality_score=72.5,
            is_anomaly=False,
        )
        result = supervisor(state)
        assert result["next_agent"] == SUMMARIZER

    def test_routes_to_reporter_after_summarizer(self):
        state = make_state(
            cleaning_complete=True,
            scoring_complete=True,
            anomaly_complete=True,
            summarizer_complete=True,
        )
        result = supervisor(state)
        assert result["next_agent"] == REPORTER

    def test_routes_to_end_when_pipeline_complete(self):
        state  = make_state(pipeline_complete=True)
        result = supervisor(state)
        assert result["next_agent"] == END

    def test_routes_to_end_on_error(self):
        state  = make_state(error="Database connection failed")
        result = supervisor(state)
        assert result["next_agent"] == END

    def test_skipped_agents_accumulate(self):
        state = make_state(data_quality_score=0.95)
        result = supervisor(state)
        assert DATA_CLEANER in result["agents_skipped"]

    def test_clean_provider_takes_shorter_path(self):
        """A provider with clean data skips the cleaner — proving dynamic routing."""
        clean_state  = make_state(data_quality_score=0.98)
        dirty_state  = make_state(data_quality_score=0.60)
        clean_result = supervisor(clean_state)
        dirty_result = supervisor(dirty_state)
        assert clean_result["next_agent"] == STATISTICAL
        assert dirty_result["next_agent"] == DATA_CLEANER


class TestInitialState:

    def test_initial_state_has_correct_npi(self):
        state = initial_state("9876543210", "run-abc")
        assert state["npi"] == "9876543210"

    def test_initial_state_starts_at_supervisor(self):
        state = initial_state("9876543210", "run-abc")
        assert state["next_agent"] == "supervisor"

    def test_initial_state_has_empty_agent_lists(self):
        state = initial_state("9876543210", "run-abc")
        assert state["agents_executed"] == []
        assert state["agents_skipped"] == []

    def test_initial_state_all_outputs_are_none(self):
        state = initial_state("9876543210", "run-abc")
        assert state["quality_score"] is None
        assert state["is_anomaly"] is None
        assert state["pipeline_complete"] is None
