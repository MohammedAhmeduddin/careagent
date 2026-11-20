"""
End-to-end tests against a real PostgreSQL: load -> LangGraph pipeline -> API.

These REPLACE the database contents, so they only run when both hold:
  - RUN_INTEGRATION=1
  - the database is local (POSTGRES_HOST localhost/127.0.0.1, no DATABASE_URL)
CI runs them against a Postgres service container.

    RUN_INTEGRATION=1 POSTGRES_SSLMODE=disable PYTHONPATH=src pytest tests/integration
"""

import os
import uuid

import pytest

from careagent.config import get_settings

settings = get_settings()
LOCAL = settings.postgres_host in ("localhost", "127.0.0.1") and not settings.database_url_env

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_INTEGRATION") == "1" and LOCAL),
    reason="Integration tests need RUN_INTEGRATION=1 and a local PostgreSQL",
)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from careagent.api.main import app  # noqa: E402
from careagent.db.models import AgentRun, PeerGroup, Provider  # noqa: E402
from careagent.db.session import SessionLocal, get_db, reset_schema  # noqa: E402
from careagent.graph.state import initial_state  # noqa: E402
from careagent.graph.workflow import careagent_graph  # noqa: E402
from careagent.scoring.load import load_dataframe  # noqa: E402
from careagent.scoring.peer import DISCLAIMER  # noqa: E402
from tests.conftest import (  # noqa: E402
    MISSING_DATA_NPI, OUTLIER_COST_NPI, SMALL_GROUP_NPI, TYPICAL_NPI, make_cms_frame,
)


@pytest.fixture(scope="module", autouse=True)
def loaded_db():
    reset_schema()
    with get_db() as db:
        counts = load_dataframe(db, make_cms_frame(), year=2024)
    return counts


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def run(npi):
    return careagent_graph.invoke(initial_state(npi=npi, run_id=str(uuid.uuid4())))


class TestLoad:
    def test_counts(self, loaded_db):
        assert loaded_db["providers"] == 105
        assert loaded_db["peer_groups"] == 2
        assert loaded_db["outlier"] == 2

    def test_rows_in_database(self):
        with SessionLocal() as db:
            assert db.get(Provider, OUTLIER_COST_NPI).billing_pattern_status == "outlier"
            assert db.get(Provider, MISSING_DATA_NPI).bene_avg_risk_score is None
            assert db.get(PeerGroup, "Cardiology").peer_count == 59

    def test_reload_replaces_instead_of_duplicating(self):
        with get_db() as db:
            load_dataframe(db, make_cms_frame(), year=2024)
        with SessionLocal() as db:
            assert len(db.scalars(select(Provider.npi)).all()) == 105


class TestGraph:
    def test_outlier_path(self):
        out = run(OUTLIER_COST_NPI)
        assert out["error"] is None and out["pipeline_complete"]
        assert out["is_outlier"] is True and out["outlier_reasons"]
        assert out["agents_executed"] == ["data_cleaner", "statistical", "anomaly",
                                          "summarizer", "reporter"]

    def test_typical_path(self):
        out = run(TYPICAL_NPI)
        assert out["billing_pattern_status"] == "typical" and out["is_outlier"] is False

    def test_small_specialty_skips_anomaly(self):
        out = run(SMALL_GROUP_NPI)
        assert out["billing_pattern_status"] == "insufficient_peers"
        assert out["agents_skipped"] == ["anomaly"]
        assert "anomaly" not in out["agents_executed"]

    def test_missing_data_skips_anomaly_and_names_fields(self):
        out = run(MISSING_DATA_NPI)
        assert out["billing_pattern_status"] == "insufficient_data"
        assert "bene_avg_risk_score" in out["missing_fields"]
        assert "bene_avg_risk_score" in out["performance_narrative"]

    def test_results_written_back(self):
        run(TYPICAL_NPI)
        with SessionLocal() as db:
            p = db.get(Provider, TYPICAL_NPI)
            assert p.performance_narrative.endswith(DISCLAIMER)
            assert p.narrative_source == "template"
            assert p.last_scored_at is not None


class TestApi:
    def test_analyze_and_agent_run_logged(self, client):
        resp = client.post("/analyze", json={"npi": OUTLIER_COST_NPI})
        assert resp.status_code == 200
        data = resp.json()
        assert data["billing_pattern_status"] == "outlier"
        assert data["peer_benchmarks"]["peer_count"] == 59
        with SessionLocal() as db:
            run_row = db.scalars(select(AgentRun).where(AgentRun.run_id == data["run_id"])).one()
            assert run_row.status == "completed" and run_row.final_status == "outlier"

    def test_unknown_npi(self, client):
        assert client.post("/analyze", json={"npi": "9999999999"}).status_code == 404

    def test_health(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["providers_loaded"] == 105
        assert data["outliers_flagged"] == 2
        assert data["peer_groups"] == 2

    def test_root_examples_are_real_npis(self, client):
        examples = client.get("/").json()["example_npis"]
        assert examples["outlier"] in (OUTLIER_COST_NPI, "1000000002")
        assert examples["insufficient_peers"].startswith("3")


class TestEdgeCasesOnPostgres:
    def test_create_tables_is_idempotent_and_adds_missing_column(self):
        from sqlalchemy import text

        from careagent.db.session import create_tables, engine
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE providers DROP COLUMN narrative_fingerprint"))
        create_tables()
        create_tables()
        with engine.connect() as conn:
            cols = {r[0] for r in conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name='providers'"))}
        assert "narrative_fingerprint" in cols

    def test_llm_narrative_is_cached_across_runs(self):
        import json
        from unittest.mock import patch

        from careagent.agents import summarizer
        good = "Cost per beneficiary is far above the Cardiology median."
        with patch.object(summarizer.settings, "openai_api_key", "sk-test"), \
             patch.object(summarizer, "_call_gpt",
                          return_value=(json.dumps({"narrative": good}), 42)) as call:
            first = run(OUTLIER_COST_NPI)
            second = run(OUTLIER_COST_NPI)
        assert call.call_count == 1
        assert first["narrative_source"] == "llm" and first["narrative_cached"] is False
        assert second["narrative_cached"] is True and second["llm_tokens_used"] == 0
        assert second["performance_narrative"] == first["performance_narrative"]

    def test_changed_model_invalidates_cache(self):
        import json
        from unittest.mock import patch

        from careagent.agents import summarizer
        with patch.object(summarizer.settings, "openai_api_key", "sk-test"), \
             patch.object(summarizer, "_call_gpt",
                          return_value=(json.dumps({"narrative": "Above the median."}), 5)) as call:
            run(TYPICAL_NPI)
            with patch.object(summarizer.settings, "openai_model", "another-model"):
                out = run(TYPICAL_NPI)
        assert call.call_count == 2 and out["narrative_cached"] is False

    def test_loader_rejects_duplicates_before_touching_data(self):
        import pandas as pd
        frame = make_cms_frame()
        bad = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="more than once"):
            with get_db() as db:
                load_dataframe(db, bad, year=2024)
        with SessionLocal() as db:
            assert len(db.scalars(select(Provider.npi)).all()) == 105  # untouched
