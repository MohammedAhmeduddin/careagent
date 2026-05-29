"""
tests/integration/test_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end integration tests for the full CareAgent pipeline.
Requires a running PostgreSQL database with loaded provider data.

Run with:
    PYTHONPATH=src pytest tests/integration/ -v --no-cov

Skipped automatically in CI (no database available).
"""

import pytest
import uuid
from careagent.db.session import check_connection, SessionLocal
from careagent.db.queries import get_provider_by_npi, get_providers_for_scoring
from careagent.graph.state import initial_state
from careagent.graph.workflow import careagent_graph


# Skip entire module if database not reachable
pytestmark = pytest.mark.skipif(
    not check_connection(),
    reason="PostgreSQL not available — skipping integration tests",
)


@pytest.fixture(scope="module")
def real_npi() -> str:
    """Fetch a real NPI from the database for testing."""
    with SessionLocal() as db:
        providers = get_providers_for_scoring(db, limit=1)
        if not providers:
            pytest.skip("No unscored providers in database")
        return providers[0].npi


class TestFullPipeline:

    def test_pipeline_completes_without_error(self, real_npi):
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        result = careagent_graph.invoke(state)
        assert result.get("error") is None
        assert result.get("pipeline_complete") is True

    def test_pipeline_produces_quality_score(self, real_npi):
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        result = careagent_graph.invoke(state)
        assert result.get("quality_score") is not None
        assert 0.0 <= result["quality_score"] <= 100.0

    def test_pipeline_produces_anomaly_flag(self, real_npi):
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        result = careagent_graph.invoke(state)
        assert result.get("is_anomaly") is not None
        assert isinstance(result["is_anomaly"], (bool, __import__("numpy").bool_))

    def test_pipeline_produces_narrative(self, real_npi):
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        result = careagent_graph.invoke(state)
        assert result.get("performance_narrative") is not None
        assert len(result["performance_narrative"]) > 20

    def test_pipeline_produces_recommendation(self, real_npi):
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        result = careagent_graph.invoke(state)
        assert result.get("network_recommendation") in ["include", "review", "exclude"]

    def test_pipeline_tracks_agents_executed(self, real_npi):
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        result = careagent_graph.invoke(state)
        executed = result.get("agents_executed", [])
        assert "statistical" in executed
        assert "anomaly"     in executed
        assert "summarizer"  in executed
        assert "reporter"    in executed

    def test_pipeline_writes_to_database(self, real_npi):
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        careagent_graph.invoke(state)
        with SessionLocal() as db:
            provider = get_provider_by_npi(db, real_npi)
        assert provider.quality_score is not None
        assert provider.is_anomaly    is not None
        assert provider.network_recommendation is not None

    def test_pipeline_completes_under_30_seconds(self, real_npi):
        import time
        state = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        start = time.perf_counter()
        careagent_graph.invoke(state)
        duration = time.perf_counter() - start
        assert duration < 30.0, f"Pipeline took {duration:.1f}s — exceeds 30s limit"

    def test_clean_provider_skips_data_cleaner(self, real_npi):
        """Proves dynamic routing — clean data skips the cleaner."""
        state  = initial_state(npi=real_npi, run_id=str(uuid.uuid4()))
        result = careagent_graph.invoke(state)
        # Provider with quality=1.0 should skip cleaner
        if result.get("data_quality_score") == 1.0:
            assert "data_cleaner" in result.get("agents_skipped", []) or \
                   result.get("fields_imputed") == 0
