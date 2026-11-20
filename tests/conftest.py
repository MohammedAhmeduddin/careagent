"""Shared test fixtures: a small CMS-format dataset with known structure."""

import numpy as np
import pandas as pd
import pytest

from careagent.scoring.peer import CMS_COLUMNS

OUTLIER_COST_NPI = "1000000001"      # Cardiology, cost per beneficiary ~6x peers
OUTLIER_CHARGE_NPI = "1000000002"    # Cardiology, charge ratio ~4x peers
MILD_HIGH_NPI = "1000000003"         # Cardiology, ~1.6x cost after noise: modified z ~2.9
MISSING_DATA_NPI = "1000000004"      # Cardiology, risk score missing
TYPICAL_NPI = "1000000010"           # Cardiology, ordinary
SMALL_GROUP_NPI = "3000000000"       # "Rare Specialty", 5 providers


def make_cms_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def add(npi, ptype, risk, cost_mult=1.0, charge_mult=1.0, base_cost=300.0):
        benes = float(rng.integers(50, 500))
        cost_pb = base_cost * (risk or 1.0) * np.exp(rng.normal(0, 0.15)) * cost_mult
        srv_pb = 8.0 * (risk or 1.0) ** 0.5 * np.exp(rng.normal(0, 0.15))
        allowed = cost_pb * benes * 1.05
        charge_ratio = 3.0 * np.exp(rng.normal(0, 0.1)) * charge_mult
        rows.append({
            "Rndrng_NPI": npi,
            "Rndrng_Prvdr_Ent_Cd": "I",
            "Rndrng_Prvdr_Type": ptype,
            "Rndrng_Prvdr_City": "Newark",
            "Rndrng_Prvdr_State_Abrvtn": "NJ",
            "Rndrng_Prvdr_Zip5": "07102",
            "Rndrng_Prvdr_RUCA": "1",
            "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "Y",
            "Tot_HCPCS_Cds": "20",
            "Tot_Benes": str(benes),
            "Tot_Srvcs": str(srv_pb * benes),
            "Tot_Sbmtd_Chrg": str(allowed * charge_ratio),
            "Tot_Mdcr_Alowd_Amt": str(allowed),
            "Tot_Mdcr_Pymt_Amt": str(allowed * 0.8),
            "Tot_Mdcr_Stdzd_Amt": str(cost_pb * benes),
            "Bene_Avg_Age": "72",
            "Bene_Avg_Risk_Scre": "" if risk is None else str(risk),
        })

    add(OUTLIER_COST_NPI, "Cardiology", 1.5, cost_mult=6.0)
    add(OUTLIER_CHARGE_NPI, "Cardiology", 1.5, charge_mult=4.0)
    add(MILD_HIGH_NPI, "Cardiology", 1.5, cost_mult=1.75)
    add(MISSING_DATA_NPI, "Cardiology", None)
    for i in range(56):
        add(f"10000000{10 + i}", "Cardiology", float(rng.uniform(0.8, 3.0)))
    for i in range(40):
        add(f"20000000{10 + i}", "Family Practice", float(rng.uniform(0.6, 2.0)), base_cost=150.0)
    for i in range(5):
        add(f"300000000{i}", "Rare Specialty", 1.2)

    df = pd.DataFrame(rows)
    assert set(CMS_COLUMNS) <= set(df.columns)
    return df


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Each test starts with an empty /analyze rate-limit window."""
    from careagent.api.ratelimit import analyze_limiter
    analyze_limiter.reset()
    yield
    analyze_limiter.reset()


@pytest.fixture
def cms_frame() -> pd.DataFrame:
    return make_cms_frame()
