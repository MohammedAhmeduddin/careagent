"""Agents and supervisor routing, with the database mocked."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from careagent.agents import anomaly, data_cleaner, reporter, statistical, summarizer
from careagent.agents.supervisor import (
    ANOMALY, DATA_CLEANER, END, REPORTER, STATISTICAL, SUMMARIZER,
    route_after_supervisor, supervisor,
)
from careagent.graph.state import initial_state
from careagent.scoring.peer import DISCLAIMER

NPI = "1234567890"


def fake_provider(**overrides):
    base = dict(
        npi=NPI, provider_type="Cardiology", city="Newark", state="NJ",
        total_beneficiaries=300.0, total_services=4000.0, total_hcpcs_codes=25.0,
        total_submitted_charges=900000.0, total_allowed_amount=300000.0,
        total_standardized_payment=280000.0, bene_avg_risk_score=1.8,
        cost_per_beneficiary=933.3, services_per_beneficiary=13.3, charge_to_allowed_ratio=3.0,
        peer_group_size=891, billing_pattern_status="outlier",
        cost_z=5.1, cost_ratio_to_peer_median=3.1, cost_percentile=99.2,
        services_z=1.0, services_ratio_to_peer_median=1.3, services_percentile=80.0,
        charge_z=0.2, charge_ratio_to_peer_median=1.05, charge_percentile=55.0,
        isolation_forest_percentile=98.7,
        outlier_reasons="Risk-adjusted Medicare cost per beneficiary is 3.1x the Cardiology median (modified z = 5.1)",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def fake_group():
    return SimpleNamespace(
        provider_type="Cardiology", peer_count=891, median_cost_per_beneficiary=301.2,
        median_services_per_beneficiary=10.2, median_charge_to_allowed_ratio=2.9,
        median_risk_score=1.7,
    )


@contextmanager
def fake_db():
    yield MagicMock()


@pytest.fixture
def state():
    return initial_state(NPI, "run-1")


class TestDataValidator:
    def test_complete_provider(self, state):
        with patch.object(data_cleaner, "get_db", fake_db), \
             patch.object(data_cleaner, "get_provider_by_npi", return_value=fake_provider()), \
             patch.object(data_cleaner, "update_provider_validation") as save:
            out = data_cleaner.data_cleaner_agent(state)
        assert out["data_completeness"] == 1.0
        assert out["missing_fields"] == []
        assert out["provider_type"] == "Cardiology" and out["city"] == "Newark"
        assert out["agents_executed"] == ["data_cleaner"]
        save.assert_called_once()

    def test_missing_and_invalid_fields_are_reported_not_imputed(self, state):
        p = fake_provider(bene_avg_risk_score=None, total_beneficiaries=0.0)
        with patch.object(data_cleaner, "get_db", fake_db), \
             patch.object(data_cleaner, "get_provider_by_npi", return_value=p), \
             patch.object(data_cleaner, "update_provider_validation"):
            out = data_cleaner.data_cleaner_agent(state)
        assert set(out["missing_fields"]) == {"bene_avg_risk_score", "total_beneficiaries"}
        assert out["data_completeness"] < 1.0
        assert "not imputed" in out["cleaning_notes"]
        assert p.bene_avg_risk_score is None  # untouched

    def test_missing_provider_type(self):
        completeness, problems = data_cleaner.validate(fake_provider(provider_type=None))
        assert problems == ["provider_type"] and completeness < 1

    def test_unknown_npi_sets_error(self, state):
        with patch.object(data_cleaner, "get_db", fake_db), \
             patch.object(data_cleaner, "get_provider_by_npi", return_value=None):
            assert "not found" in data_cleaner.data_cleaner_agent(state)["error"]


class TestStatistical:
    def test_reads_stored_comparison_and_benchmarks(self, state):
        with patch.object(statistical, "get_db", fake_db), \
             patch.object(statistical, "get_provider_by_npi", return_value=fake_provider()), \
             patch.object(statistical, "get_peer_group", return_value=fake_group()):
            out = statistical.statistical_agent(state)
        assert out["billing_pattern_status"] == "outlier"
        assert out["peer_group_size"] == 891
        assert out["peer_comparison"]["cost"]["ratio_to_peer_median"] == 3.1
        assert out["peer_comparison"]["cost"]["modified_z"] == 5.1
        assert out["peer_benchmarks"]["median_cost_per_beneficiary"] == 301.2
        assert out["provider_metrics"]["cost_per_beneficiary"] == 933.3
        assert out["scoring_complete"] is True

    def test_no_peer_group_gives_null_benchmarks(self, state):
        p = fake_provider(billing_pattern_status="insufficient_peers", cost_z=None,
                          cost_ratio_to_peer_median=None, cost_percentile=None)
        with patch.object(statistical, "get_db", fake_db), \
             patch.object(statistical, "get_provider_by_npi", return_value=p), \
             patch.object(statistical, "get_peer_group", return_value=None):
            out = statistical.statistical_agent(state)
        assert out["peer_benchmarks"] is None
        assert out["peer_comparison"]["cost"]["ratio_to_peer_median"] is None

    def test_unknown_npi_sets_error(self, state):
        with patch.object(statistical, "get_db", fake_db), \
             patch.object(statistical, "get_provider_by_npi", return_value=None):
            assert statistical.statistical_agent(state)["error"]


class TestAnomaly:
    def test_outlier_with_reasons(self, state):
        with patch.object(anomaly, "get_db", fake_db), \
             patch.object(anomaly, "get_provider_by_npi", return_value=fake_provider(
                 outlier_reasons="Reason A; Reason B")):
            out = anomaly.anomaly_agent(state)
        assert out["is_outlier"] is True
        assert out["outlier_reasons"] == ["Reason A", "Reason B"]
        assert out["isolation_forest_percentile"] == 98.7

    def test_typical_has_no_reasons(self, state):
        with patch.object(anomaly, "get_db", fake_db), \
             patch.object(anomaly, "get_provider_by_npi", return_value=fake_provider(
                 billing_pattern_status="typical", outlier_reasons=None)):
            out = anomaly.anomaly_agent(state)
        assert out["is_outlier"] is False and out["outlier_reasons"] == []

    def test_unknown_npi_sets_error(self, state):
        with patch.object(anomaly, "get_db", fake_db), \
             patch.object(anomaly, "get_provider_by_npi", return_value=None):
            assert anomaly.anomaly_agent(state)["error"]


class TestReporter:
    def test_scorecard_has_status_and_disclaimer(self, state):
        state = {**state, "billing_pattern_status": "typical", "provider_type": "Cardiology"}
        with patch.object(reporter, "get_db", fake_db), \
             patch.object(reporter, "update_provider_report") as save:
            out = reporter.reporter_agent(state)
        assert out["pipeline_complete"] is True
        assert out["scorecard"]["billing_pattern_status"] == "typical"
        assert out["scorecard"]["disclaimer"] == DISCLAIMER
        assert "network_recommendation" not in out["scorecard"]
        save.assert_called_once()


class TestSupervisor:
    def test_full_route_for_comparable_provider(self, state):
        assert supervisor(state)["next_agent"] == DATA_CLEANER
        s = {**state, "cleaning_complete": True}
        assert supervisor(s)["next_agent"] == STATISTICAL
        s = {**s, "scoring_complete": True, "billing_pattern_status": "outlier"}
        assert supervisor(s)["next_agent"] == ANOMALY
        s = {**s, "anomaly_complete": True}
        assert supervisor(s)["next_agent"] == SUMMARIZER
        s = {**s, "summarizer_complete": True}
        assert supervisor(s)["next_agent"] == REPORTER
        s = {**s, "pipeline_complete": True}
        assert supervisor(s)["next_agent"] == END

    @pytest.mark.parametrize("status", ["insufficient_peers", "insufficient_data"])
    def test_anomaly_skipped_when_nothing_to_compare(self, state, status):
        s = {**state, "cleaning_complete": True, "scoring_complete": True,
             "billing_pattern_status": status}
        out = supervisor(s)
        assert out["next_agent"] == SUMMARIZER
        assert out["agents_skipped"] == [ANOMALY]
        assert out["anomaly_complete"] is True and out["is_outlier"] is None

    def test_error_stops(self, state):
        assert supervisor({**state, "error": "boom"})["next_agent"] == END

    def test_route_helper(self, state):
        assert route_after_supervisor({**state, "next_agent": ANOMALY}) == ANOMALY
        assert route_after_supervisor({}) == END


class TestSummarizerTemplate:
    def comparable_state(self, state, **extra):
        return {**state, "provider_type": "Cardiology", "city": "Newark", "state": "NJ",
                "billing_pattern_status": "outlier", "peer_group_size": 891,
                "provider_metrics": {"beneficiaries": 300.0, "avg_patient_risk_score": 1.8},
                "peer_comparison": {
                    "cost": {"label": "risk-adjusted Medicare cost per beneficiary",
                             "ratio_to_peer_median": 3.1, "percentile": 99.2, "modified_z": 5.1},
                    "services": {"label": "risk-adjusted services per beneficiary",
                                 "ratio_to_peer_median": 1.3, "percentile": 80.0, "modified_z": 1.0},
                    "charge": {"label": "submitted-charge to allowed-amount ratio",
                               "ratio_to_peer_median": None, "percentile": None, "modified_z": None},
                },
                "outlier_reasons": ["Risk-adjusted Medicare cost per beneficiary is 3.1x the Cardiology median (modified z = 5.1)"],
                **extra}

    def test_outlier_template(self, state):
        text = summarizer._template_narrative(self.comparable_state(state))
        assert "891 Cardiology providers" in text and "3.1x" in text and "outlier" in text

    def test_typical_template(self, state):
        text = summarizer._template_narrative(self.comparable_state(state, outlier_reasons=[]))
        assert "within the typical range" in text

    def test_insufficient_peers_template(self, state):
        s = {**state, "provider_type": "Rare", "billing_pattern_status": "insufficient_peers",
             "peer_group_size": 4}
        assert "4 providers" in summarizer._template_narrative(s)

    def test_insufficient_data_template(self, state):
        s = {**state, "billing_pattern_status": "insufficient_data",
             "missing_fields": ["bene_avg_risk_score"]}
        assert "bene_avg_risk_score" in summarizer._template_narrative(s)

    def test_prompt_contains_facts_and_rules_but_no_npi(self, state):
        prompt = summarizer._build_prompt(self.comparable_state(state))
        assert "Cardiology" in prompt and "3.1x" in prompt
        assert "do not suggest fraud" in prompt.lower()
        assert NPI not in prompt

    def test_facts_skip_metrics_without_values(self, state):
        facts = summarizer._facts(self.comparable_state(state))
        assert not any("submitted-charge" in f for f in facts)
