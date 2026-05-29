"""
tests/unit/test_agents_week3.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Statistical and Anomaly agent logic.
Uses plain dataclasses as test doubles — no SQLAlchemy instrumentation needed
for pure logic tests.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from careagent.db.models import Base, Provider
from careagent.agents.statistical import _compute_quality_score, _compute_percentile_rank
from careagent.agents.anomaly import _build_feature_matrix, _explain_anomaly


# ── Test double — plain dataclass, no SQLAlchemy ───────────────────────────────

@dataclass
class FakeProvider:
    """Lightweight stand-in for Provider ORM object in pure logic tests."""
    npi: str = "1234567890"
    provider_type: str = "Internal Medicine"
    avg_medicare_payment: Optional[float] = 150.0
    avg_submitted_charge: Optional[float] = 300.0
    avg_allowed_amount: Optional[float] = 160.0
    total_services: Optional[float] = 500.0


# ── DB fixture — only needed for percentile rank tests ─────────────────────────

@pytest.fixture(scope="function")
def db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


@pytest.fixture
def sample_providers(db):
    """Insert 10 providers for percentile rank testing."""
    providers = [
        Provider(
            npi=str(1000000000 + i),
            entity_type="I",
            last_name_or_org=f"Provider{i}",
            provider_type="Internal Medicine",
            medicare_participation=True,
            total_services=float(100 * (i + 1)),
            avg_medicare_payment=float(100 + i * 10),
            avg_submitted_charge=float(200 + i * 20),
            avg_allowed_amount=float(120 + i * 10),
            cms_data_year=2022,
        )
        for i in range(10)
    ]
    db.add_all(providers)
    db.commit()
    return providers


# ── Statistical Agent tests ────────────────────────────────────────────────────

class TestQualityScoreFormula:

    def test_score_is_between_0_and_100(self):
        p = FakeProvider()
        benchmarks = {"avg_payment": 150.0, "avg_charge": 300.0}
        score = _compute_quality_score(p, benchmarks, volume_percentile=50.0)
        assert 0.0 <= score <= 100.0

    def test_efficient_provider_scores_higher(self):
        efficient   = FakeProvider(avg_submitted_charge=200.0)
        inefficient = FakeProvider(avg_submitted_charge=600.0)
        benchmarks  = {"avg_payment": 150.0, "avg_charge": 300.0}
        score_eff   = _compute_quality_score(efficient,   benchmarks, 60.0)
        score_ineff = _compute_quality_score(inefficient, benchmarks, 60.0)
        assert score_eff > score_ineff

    def test_high_volume_provider_scores_higher(self):
        p          = FakeProvider()
        benchmarks = {"avg_payment": 150.0, "avg_charge": 300.0}
        high_vol   = _compute_quality_score(p, benchmarks, volume_percentile=90.0)
        low_vol    = _compute_quality_score(p, benchmarks, volume_percentile=10.0)
        assert high_vol > low_vol

    def test_missing_fields_returns_default(self):
        p = FakeProvider(
            avg_medicare_payment=None,
            avg_submitted_charge=None,
            avg_allowed_amount=None,
        )
        score = _compute_quality_score(p, {}, volume_percentile=50.0)
        assert score == 50.0

    def test_score_with_empty_benchmarks(self):
        p = FakeProvider()
        score = _compute_quality_score(p, {}, volume_percentile=50.0)
        assert 0.0 <= score <= 100.0


class TestPercentileRank:

    def test_highest_provider_near_100th_percentile(self, db, sample_providers):
        top_npi = sample_providers[-1].npi
        pct = _compute_percentile_rank(db, top_npi, "total_services", "Internal Medicine")
        assert pct >= 80.0

    def test_lowest_provider_near_0th_percentile(self, db, sample_providers):
        bottom_npi = sample_providers[0].npi
        pct = _compute_percentile_rank(db, bottom_npi, "total_services", "Internal Medicine")
        assert pct <= 20.0

    def test_returns_50_for_missing_value(self, db, sample_providers):
        p = Provider(
            npi="9999999999",
            entity_type="I",
            last_name_or_org="Missing",
            provider_type="Internal Medicine",
            medicare_participation=True,
            total_services=None,
            cms_data_year=2022,
        )
        db.add(p)
        db.commit()
        pct = _compute_percentile_rank(db, "9999999999", "total_services", "Internal Medicine")
        assert pct == 50.0


# ── Anomaly Agent tests ────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_matrix_shape_correct(self):
        providers = [
            FakeProvider("0001", avg_submitted_charge=300.0, avg_medicare_payment=150.0,
                         avg_allowed_amount=160.0, total_services=500.0),
            FakeProvider("0002", avg_submitted_charge=400.0, avg_medicare_payment=180.0,
                         avg_allowed_amount=190.0, total_services=800.0),
            FakeProvider("0003", avg_submitted_charge=250.0, avg_medicare_payment=120.0,
                         avg_allowed_amount=130.0, total_services=300.0),
        ]
        X, npis = _build_feature_matrix(providers)
        assert X.shape == (3, 4)
        assert npis == ["0001", "0002", "0003"]

    def test_none_values_become_zero(self):
        p = FakeProvider(
            avg_submitted_charge=None,
            avg_medicare_payment=None,
            avg_allowed_amount=None,
            total_services=None,
        )
        X, _ = _build_feature_matrix([p])
        assert np.all(X == 0.0)

    def test_npi_order_preserved(self):
        providers = [
            FakeProvider("AAA", avg_submitted_charge=300.0, avg_medicare_payment=150.0,
                         avg_allowed_amount=160.0, total_services=500.0),
            FakeProvider("BBB", avg_submitted_charge=400.0, avg_medicare_payment=180.0,
                         avg_allowed_amount=190.0, total_services=800.0),
        ]
        _, npis = _build_feature_matrix(providers)
        assert npis == ["AAA", "BBB"]


class TestExplainAnomaly:

    def test_non_anomaly_returns_normal_message(self):
        p = FakeProvider()
        result = _explain_anomaly(p, score=-0.05, is_anomaly=False)
        assert "normal range" in result.lower()

    def test_anomaly_mentions_high_cost(self):
        p = FakeProvider(avg_submitted_charge=900.0, avg_medicare_payment=150.0)
        result = _explain_anomaly(p, score=-0.45, is_anomaly=True)
        assert "anomaly" in result.lower()

    def test_anomaly_includes_score(self):
        p = FakeProvider()
        result = _explain_anomaly(p, score=-0.312, is_anomaly=True)
        assert "-0.312" in result or "0.312" in result

    def test_non_anomaly_includes_score(self):
        p = FakeProvider()
        result = _explain_anomaly(p, score=-0.05, is_anomaly=False)
        assert "-0.05" in result or "0.05" in result
