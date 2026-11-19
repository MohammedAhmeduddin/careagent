"""API routes and schemas, with the pipeline and database mocked."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from careagent.api.main import app
from careagent.api.schemas import AnalyzeRequest
from careagent.scoring.peer import DISCLAIMER

client = TestClient(app)
NPI = "1234567890"


@contextmanager
def fake_db():
    yield MagicMock()


def final_state(**extra):
    return {
        "npi": NPI, "run_id": "r", "error": None, "pipeline_complete": True,
        "provider_type": "Cardiology", "city": "Newark", "state": "NJ",
        "billing_pattern_status": "outlier", "peer_group_size": 891,
        "provider_metrics": {"beneficiaries": 300.0},
        "peer_benchmarks": {"peer_count": 891},
        "peer_comparison": {"cost": {"label": "cost", "ratio_to_peer_median": 3.1,
                                     "percentile": 99.2, "modified_z": 5.1}},
        "is_outlier": True, "outlier_reasons": ["Reason"],
        "isolation_forest_percentile": 98.0, "data_completeness": 1.0, "missing_fields": [],
        "performance_narrative": "Text.", "narrative_source": "llm", "llm_tokens_used": 250,
        "scorecard_version": "2.0.0",
        "agents_executed": ["data_cleaner", "statistical", "anomaly", "summarizer", "reporter"],
        "agents_skipped": [], **extra,
    }


@pytest.fixture
def pipeline():
    with patch("careagent.api.routes.analyze.get_db", fake_db), \
         patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=object()), \
         patch("careagent.api.routes.analyze.create_agent_run"), \
         patch("careagent.api.routes.analyze.complete_agent_run") as complete, \
         patch("careagent.api.routes.analyze.fail_agent_run") as fail, \
         patch("careagent.api.routes.analyze.careagent_graph") as graph:
        graph.invoke.return_value = final_state()
        yield SimpleMocks(graph, complete, fail)


class SimpleMocks:
    def __init__(self, graph, complete, fail):
        self.graph, self.complete, self.fail = graph, complete, fail


class TestAnalyze:
    def test_success_returns_scorecard_with_disclaimer(self, pipeline):
        resp = client.post("/analyze", json={"npi": NPI})
        assert resp.status_code == 200
        data = resp.json()
        assert data["billing_pattern_status"] == "outlier"
        assert data["peer_comparison"]["cost"]["ratio_to_peer_median"] == 3.1
        assert data["disclaimer"] == DISCLAIMER
        assert "provider_name" not in data

    def test_run_records_tokens_status_and_llm_call(self, pipeline):
        client.post("/analyze", json={"npi": NPI})
        kwargs = pipeline.complete.call_args.kwargs
        assert kwargs["tokens_used"] == 250 and kwargs["llm_calls"] == 1
        assert kwargs["final_status"] == "outlier"

    def test_template_is_not_counted_as_llm_call(self, pipeline):
        pipeline.graph.invoke.return_value = final_state(narrative_source="template", llm_tokens_used=0)
        client.post("/analyze", json={"npi": NPI})
        assert pipeline.complete.call_args.kwargs["llm_calls"] == 0

    def test_unknown_npi_404(self):
        with patch("careagent.api.routes.analyze.get_db", fake_db), \
             patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=None):
            assert client.post("/analyze", json={"npi": NPI}).status_code == 404

    @pytest.mark.parametrize("bad", ["123", "12345678901", "abcdefghij", ""])
    def test_invalid_npi_422(self, bad):
        assert client.post("/analyze", json={"npi": bad}).status_code == 422

    def test_pipeline_exception_details_are_not_leaked(self, pipeline):
        pipeline.graph.invoke.side_effect = RuntimeError("host=db.internal password=secret")
        resp = client.post("/analyze", json={"npi": NPI})
        assert resp.status_code == 500
        assert "secret" not in resp.text and "db.internal" not in resp.text
        assert "secret" in pipeline.fail.call_args.kwargs["error_message"]

    def test_agent_error_returns_500(self, pipeline):
        pipeline.graph.invoke.return_value = final_state(error="Provider vanished")
        resp = client.post("/analyze", json={"npi": NPI})
        assert resp.status_code == 500 and "vanished" not in resp.text


class TestHealthAndRoot:
    def test_health_healthy(self):
        stats = {"total_providers": 39694, "providers_compared": 39471,
                 "outliers_flagged": 1461, "peer_groups": 71, "total_agent_runs": 0}
        with patch("careagent.api.routes.health.check_connection", return_value=True), \
             patch("careagent.api.routes.health.get_db", fake_db), \
             patch("careagent.api.routes.health.get_database_stats", return_value=stats):
            data = client.get("/health").json()
        assert data["status"] == "healthy" and data["outliers_flagged"] == 1461

    def test_health_degraded(self):
        with patch("careagent.api.routes.health.check_connection", return_value=False):
            data = client.get("/health").json()
        assert data["status"] == "degraded" and data["providers_loaded"] == 0

    def test_root_lists_example_npis(self):
        with patch("careagent.api.main.get_db", fake_db), \
             patch("careagent.api.main.get_example_npis", return_value={"typical": NPI}):
            data = client.get("/").json()
        assert data["example_npis"] == {"typical": NPI}
        assert data["disclaimer"] == DISCLAIMER

    def test_root_survives_database_outage(self):
        with patch("careagent.api.main.get_db", side_effect=RuntimeError("down")):
            resp = client.get("/")
        assert resp.status_code == 200 and resp.json()["example_npis"] == {}


class TestSchemas:
    def test_valid_npi(self):
        assert AnalyzeRequest(npi=NPI).npi == NPI

    @pytest.mark.parametrize("bad", ["123", "12345678901", "12345abcde", " 1234567890"])
    def test_invalid_npi(self, bad):
        with pytest.raises(ValidationError):
            AnalyzeRequest(npi=bad)
