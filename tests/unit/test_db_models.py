"""
tests/unit/test_db_models.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for SQLAlchemy models and query functions.
Uses SQLite in-memory — no PostgreSQL needed.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from careagent.db.models import AgentRun, Base, Provider, ProviderService
from careagent.db.queries import (
    complete_agent_run,
    create_agent_run,
    get_database_stats,
    get_provider_by_npi,
    get_providers_for_scoring,
    get_specialty_benchmarks,
    update_provider_anomaly,
    update_provider_cleaning,
    update_provider_quality_scores,
)


@pytest.fixture(scope="function")
def db() -> Session:
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


@pytest.fixture
def sample_provider(db: Session) -> Provider:
    provider = Provider(
        npi="1234567890",
        entity_type="I",
        last_name_or_org="Smith",
        first_name="John",
        credentials="MD",
        gender="M",
        provider_type="Internal Medicine",
        medicare_participation=True,
        state="NY",
        city="New York",
        zip_code="10001",
        country="US",
        total_services=1250.0,
        total_unique_beneficiaries=340.0,
        distinct_procedure_count=8,
        avg_medicare_payment=145.50,
        avg_submitted_charge=280.00,
        avg_allowed_amount=160.00,
        cms_data_year=2022,
    )
    db.add(provider)
    db.commit()
    return provider


class TestProviderModel:
    def test_provider_can_be_created(self, sample_provider):
        assert sample_provider.npi == "1234567890"

    def test_provider_repr(self, sample_provider):
        assert "1234567890" in repr(sample_provider)

    def test_defaults_to_no_quality_score(self, sample_provider):
        assert sample_provider.quality_score is None

    def test_defaults_to_no_anomaly(self, sample_provider):
        assert sample_provider.is_anomaly is None

    def test_defaults_to_no_recommendation(self, sample_provider):
        assert sample_provider.network_recommendation is None


class TestProviderServiceModel:
    def test_service_links_to_provider(self, db, sample_provider):
        service = ProviderService(
            npi=sample_provider.npi,
            hcpcs_code="99213",
            hcpcs_description="Office visit",
            place_of_service="O",
            line_service_count=450.0,
            beneficiary_unique_count=210.0,
            avg_medicare_payment_amt=78.50,
            avg_submitted_charge_amt=150.00,
            cms_data_year=2022,
        )
        db.add(service)
        db.commit()
        loaded = db.get(Provider, sample_provider.npi)
        assert len(loaded.services) == 1
        assert loaded.services[0].hcpcs_code == "99213"


class TestGetProviderByNpi:
    def test_returns_provider_when_exists(self, db, sample_provider):
        result = get_provider_by_npi(db, "1234567890")
        assert result is not None
        assert result.npi == "1234567890"

    def test_returns_none_when_not_found(self, db):
        result = get_provider_by_npi(db, "9999999999")
        assert result is None


class TestGetProvidersForScoring:
    def test_returns_unscored_providers(self, db, sample_provider):
        results = get_providers_for_scoring(db, limit=10)
        assert len(results) == 1

    def test_excludes_already_scored(self, db, sample_provider):
        update_provider_quality_scores(
            db, sample_provider.npi,
            quality_score=75.0, quality_percentile=62.0,
            cost_efficiency_ratio=1.93, volume_percentile=70.0,
        )
        db.commit()
        results = get_providers_for_scoring(db, limit=10)
        assert len(results) == 0


class TestGetSpecialtyBenchmarks:
    def test_returns_benchmarks_for_known_specialty(self, db, sample_provider):
        benchmarks = get_specialty_benchmarks(db, "Internal Medicine")
        assert "avg_payment" in benchmarks
        assert benchmarks["avg_payment"] == pytest.approx(145.50)

    def test_returns_empty_for_unknown_specialty(self, db):
        benchmarks = get_specialty_benchmarks(db, "UnknownSpecialty")
        assert benchmarks == {}


class TestUpdateFunctions:
    def test_update_quality_scores(self, db, sample_provider):
        update_provider_quality_scores(
            db, sample_provider.npi,
            quality_score=82.5, quality_percentile=78.0,
            cost_efficiency_ratio=1.93, volume_percentile=65.0,
        )
        db.commit()
        loaded = get_provider_by_npi(db, sample_provider.npi)
        assert loaded.quality_score == pytest.approx(82.5)

    def test_update_anomaly_flag(self, db, sample_provider):
        update_provider_anomaly(
            db, sample_provider.npi,
            is_anomaly=True, anomaly_score=-0.15,
            anomaly_reason="High cost relative to quality",
        )
        db.commit()
        loaded = get_provider_by_npi(db, sample_provider.npi)
        assert loaded.is_anomaly is True
        assert loaded.anomaly_score == pytest.approx(-0.15)

    def test_update_cleaning_metadata(self, db, sample_provider):
        update_provider_cleaning(
            db, sample_provider.npi,
            data_quality_score=0.88, fields_imputed=2,
            cleaning_notes="Imputed readmission_rate and volume_percentile",
        )
        db.commit()
        loaded = get_provider_by_npi(db, sample_provider.npi)
        assert loaded.data_quality_score == pytest.approx(0.88)
        assert loaded.fields_imputed == 2


class TestAgentRunQueries:
    def test_create_and_complete_run(self, db, sample_provider):
        run = create_agent_run(db, run_id="test-run-001", npi=sample_provider.npi)
        assert run.status == "running"
        complete_agent_run(
            db, run_id="test-run-001",
            agents_executed=["DataCleaner", "Statistical", "Anomaly", "Summarizer", "Reporter"],
            agents_skipped=[],
            duration_seconds=6.2, tokens_used=843,
            llm_calls=1, recommendation="include",
        )
        db.commit()
        loaded = db.query(AgentRun).filter_by(run_id="test-run-001").first()
        assert loaded.status == "completed"
        assert loaded.total_duration_seconds == pytest.approx(6.2)


class TestDatabaseStats:
    def test_stats_on_empty_database(self, db):
        stats = get_database_stats(db)
        assert stats["total_providers"] == 0
        assert stats["anomalies_flagged"] == 0

    def test_stats_reflect_loaded_data(self, db, sample_provider):
        update_provider_anomaly(
            db, sample_provider.npi,
            is_anomaly=True, anomaly_score=-0.12,
            anomaly_reason="Cost outlier",
        )
        db.commit()
        stats = get_database_stats(db)
        assert stats["total_providers"] == 1
        assert stats["anomalies_flagged"] == 1
