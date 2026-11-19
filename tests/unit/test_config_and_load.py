"""Settings (connection string handling) and DataFrame -> row conversion."""

import math

import numpy as np
import pandas as pd
import pytest

from careagent.config import Settings
from careagent.scoring.load import _clean, peer_group_rows, provider_rows
from careagent.scoring.peer import score_providers
from tests.conftest import MISSING_DATA_NPI, OUTLIER_COST_NPI


class TestDatabaseUrl:
    def test_built_from_parts_with_sslmode(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        s = Settings(_env_file=None, postgres_host="h", postgres_sslmode="disable")
        assert s.database_url.startswith("postgresql+psycopg2://")
        assert "@h:5432/" in s.database_url and s.database_url.endswith("sslmode=disable")

    def test_default_sslmode_is_require(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_SSLMODE", raising=False)
        assert Settings(_env_file=None).database_url.endswith("sslmode=require")

    @pytest.mark.parametrize("given", [
        "postgres://u:p@neon.tech/db?sslmode=require",
        "postgresql://u:p@neon.tech/db?sslmode=require",
    ])
    def test_database_url_env_overrides_and_gets_driver(self, monkeypatch, given):
        monkeypatch.setenv("DATABASE_URL", given)
        url = Settings(_env_file=None).database_url
        assert url == "postgresql+psycopg2://u:p@neon.tech/db?sslmode=require"

    def test_database_url_with_driver_kept(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "  postgresql+psycopg2://u:p@h/db ")
        assert Settings(_env_file=None).database_url == "postgresql+psycopg2://u:p@h/db"


class TestClean:
    @pytest.mark.parametrize("value", [None, float("nan"), np.float64("nan"), pd.NA])
    def test_missing_becomes_none(self, value):
        assert _clean(value) is None

    def test_numpy_scalars_become_python(self):
        assert type(_clean(np.int64(3))) is int
        assert type(_clean(np.float64(2.5))) is float
        assert _clean("x") == "x"


class TestRows:
    def test_provider_rows(self, cms_frame):
        scored, groups = score_providers(cms_frame)
        rows = {r["npi"]: r for r in provider_rows(scored, 2024)}
        assert len(rows) == len(cms_frame)
        out = rows[OUTLIER_COST_NPI]
        assert out["billing_pattern_status"] == "outlier"
        assert out["cms_data_year"] == 2024
        assert isinstance(out["peer_group_size"], int)
        missing = rows[MISSING_DATA_NPI]
        assert missing["bene_avg_risk_score"] is None
        assert missing["outlier_reasons"] is None
        for row in rows.values():
            for value in row.values():
                assert not (isinstance(value, float) and math.isnan(value))

    def test_peer_group_rows(self, cms_frame):
        _, groups = score_providers(cms_frame)
        rows = peer_group_rows(groups, 2024)
        assert {r["provider_type"] for r in rows} == {"Cardiology", "Family Practice"}
        assert all(r["cms_data_year"] == 2024 for r in rows)

    def test_peer_group_rows_empty(self):
        assert peer_group_rows(pd.DataFrame(), 2024) == []
