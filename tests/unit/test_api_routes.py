"""
tests/unit/test_api_routes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FastAPI route tests using TestClient + mocked pipeline.
Covers /health and /analyze endpoints without hitting real DB or agents.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, UTC
from fastapi.testclient import TestClient
from careagent.api.main import app

client = TestClient(app)


# ── Mock data ──────────────────────────────────────────────────────────────────

def mock_provider():
    p = MagicMock()
    p.npi              = "1234567890"
    p.last_name_or_org = "Smith"
    p.first_name       = "John"
    p.provider_type    = "Internal Medicine"
    p.state            = "NY"
    return p


def mock_final_state():
    return {
        "npi":                    "1234567890",
        "run_id":                 "test-run-001",
        "error":                  None,
        "pipeline_complete":      True,
        "quality_score":          74.3,
        "quality_percentile":     62.0,
        "cost_efficiency_ratio":  1.93,
        "volume_percentile":      55.0,
        "is_anomaly":             False,
        "anomaly_score":          -0.05,
        "anomaly_reason":         "Within normal range",
        "data_quality_score":     1.0,
        "fields_imputed":         0,
        "performance_narrative":  "Provider performs above average.",
        "narrative_faithfulness": None,
        "narrative_relevancy":    None,
        "network_recommendation": "include",
        "scorecard_version":      "0.1.0",
        "agents_executed":        ["data_cleaner", "statistical", "anomaly", "summarizer", "reporter"],
        "agents_skipped":         [],
        "scorecard":              {},
    }


# ── Health endpoint tests ──────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200_when_db_connected(self):
        with patch("careagent.api.routes.health.check_connection", return_value=True), \
             patch("careagent.api.routes.health.get_db") as mock_db, \
             patch("careagent.api.routes.health.get_database_stats") as mock_stats:

            mock_stats.return_value = {
                "total_providers":  10000,
                "providers_scored": 5000,
                "anomalies_flagged": 312,
                "total_agent_runs": 150,
                "completed_runs":   148,
            }
            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)

            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["providers_loaded"] == 10000

    def test_health_returns_degraded_when_db_unreachable(self):
        with patch("careagent.api.routes.health.check_connection", return_value=False):
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["database"] == "unreachable"
            assert data["providers_loaded"] == 0

    def test_health_includes_version(self):
        with patch("careagent.api.routes.health.check_connection", return_value=False):
            resp = client.get("/health")
            assert resp.json()["version"] == "0.1.0"


# ── Analyze endpoint tests ─────────────────────────────────────────────────────

class TestAnalyzeEndpoint:

    def test_analyze_returns_200_for_valid_npi(self):
        with patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=mock_provider()), \
             patch("careagent.api.routes.analyze.get_db") as mock_db, \
             patch("careagent.api.routes.analyze.create_agent_run"), \
             patch("careagent.api.routes.analyze.careagent_graph") as mock_graph, \
             patch("careagent.api.routes.analyze.complete_agent_run"), \
             patch("careagent.api.routes.analyze.fail_agent_run"):

            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            mock_graph.invoke.return_value = mock_final_state()

            resp = client.post("/analyze", json={"npi": "1234567890"})
            assert resp.status_code == 200

    def test_analyze_returns_correct_scorecard_fields(self):
        with patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=mock_provider()), \
             patch("careagent.api.routes.analyze.get_db") as mock_db, \
             patch("careagent.api.routes.analyze.create_agent_run"), \
             patch("careagent.api.routes.analyze.careagent_graph") as mock_graph, \
             patch("careagent.api.routes.analyze.complete_agent_run"), \
             patch("careagent.api.routes.analyze.fail_agent_run"):

            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            mock_graph.invoke.return_value = mock_final_state()

            resp = client.post("/analyze", json={"npi": "1234567890"})
            data = resp.json()
            assert data["npi"]                    == "1234567890"
            assert data["quality_score"]          == 74.3
            assert data["network_recommendation"] == "include"
            assert data["is_anomaly"]             is False
            assert "agents_executed" in data

    def test_analyze_returns_404_for_unknown_npi(self):
        with patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=None), \
             patch("careagent.api.routes.analyze.get_db") as mock_db:

            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)

            resp = client.post("/analyze", json={"npi": "9999999999"})
            assert resp.status_code == 404

    def test_analyze_returns_422_for_invalid_npi(self):
        resp = client.post("/analyze", json={"npi": "short"})
        assert resp.status_code == 422

    def test_analyze_returns_422_for_non_numeric_npi(self):
        resp = client.post("/analyze", json={"npi": "123456789A"})
        assert resp.status_code == 422

    def test_analyze_returns_500_on_pipeline_error(self):
        with patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=mock_provider()), \
             patch("careagent.api.routes.analyze.get_db") as mock_db, \
             patch("careagent.api.routes.analyze.create_agent_run"), \
             patch("careagent.api.routes.analyze.careagent_graph") as mock_graph, \
             patch("careagent.api.routes.analyze.fail_agent_run"):

            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            mock_graph.invoke.side_effect  = Exception("Database timeout")

            resp = client.post("/analyze", json={"npi": "1234567890"})
            assert resp.status_code == 500

    def test_analyze_returns_500_on_pipeline_state_error(self):
        error_state = {**mock_final_state(), "error": "Provider not found"}
        with patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=mock_provider()), \
             patch("careagent.api.routes.analyze.get_db") as mock_db, \
             patch("careagent.api.routes.analyze.create_agent_run"), \
             patch("careagent.api.routes.analyze.careagent_graph") as mock_graph, \
             patch("careagent.api.routes.analyze.fail_agent_run"):

            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            mock_graph.invoke.return_value = error_state

            resp = client.post("/analyze", json={"npi": "1234567890"})
            assert resp.status_code == 500

    def test_analyze_response_includes_pipeline_duration(self):
        with patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=mock_provider()), \
             patch("careagent.api.routes.analyze.get_db") as mock_db, \
             patch("careagent.api.routes.analyze.create_agent_run"), \
             patch("careagent.api.routes.analyze.careagent_graph") as mock_graph, \
             patch("careagent.api.routes.analyze.complete_agent_run"), \
             patch("careagent.api.routes.analyze.fail_agent_run"):

            mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db.return_value.__exit__  = MagicMock(return_value=False)
            mock_graph.invoke.return_value = mock_final_state()

            resp = client.post("/analyze", json={"npi": "1234567890"})
            assert resp.json()["pipeline_duration_seconds"] is not None
