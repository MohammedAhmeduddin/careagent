"""
careagent.scoring.peer
~~~~~~~~~~~~~~~~~~~~~~
Peer-group comparison of Medicare billing patterns.

Input: one row per provider from the CMS "Medicare Physician & Other
Practitioners - by Provider" dataset. Output: per-provider comparison
against providers of the same specialty, plus one summary row per specialty.

What this measures
    How a provider's Medicare billing compares with peers in the same
    specialty, after adjusting for how sick their patients are. It does
    NOT measure quality of care (the data has no outcomes) and an outlier
    is not evidence of fraud or wrongdoing. Every scorecard says so.

Method (all per specialty, fitted once at load time)
    1. Three per-provider metrics:
         cost_per_beneficiary      = standardized Medicare payment / beneficiaries
                                     (CMS standardization removes geographic pay
                                     differences)
         services_per_beneficiary  = services / beneficiaries
         charge_to_allowed_ratio   = submitted charges / Medicare allowed amount
    2. Risk adjustment for the first two: within the specialty, fit
       log(metric) ~ log(average HCC risk score) by least squares. The
       observed / expected (O/E) ratio removes the part explained by
       patient risk (skipped if every peer has the same risk score).
       Charges are a pricing choice, so they are not adjusted.
    3. Robust z-score of log(O/E) against the specialty median, using the
       median absolute deviation (Iglewicz & Hoaglin modified z-score).
    4. A provider is a billing-pattern outlier when, for at least one
       metric, modified z > 3.5 (statistical rule) AND the value is at
       least 2x the specialty median (practical significance). The second
       condition stops tight specialties from flagging small differences.
    5. Isolation Forest per specialty on the three log ratios gives a
       multivariate "how unusual overall" percentile. It ranks; it does
       not decide the flag, so the flag stays explainable.

Specialties with fewer than MIN_PEERS scorable providers are not scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MIN_PEERS = 30
Z_THRESHOLD = 3.5
MIN_RATIO = 2.0
RANDOM_STATE = 42

STATUS_TYPICAL = "typical"
STATUS_OUTLIER = "outlier"
STATUS_INSUFFICIENT_PEERS = "insufficient_peers"
STATUS_INSUFFICIENT_DATA = "insufficient_data"

DISCLAIMER = (
    "Compares Medicare billing patterns with same-specialty peers after "
    "adjusting for patient risk. It is not a measure of quality of care, "
    "and an outlier is not evidence of fraud or wrongdoing."
)

# CMS column -> internal name
CMS_COLUMNS = {
    "Rndrng_NPI": "npi",
    "Rndrng_Prvdr_Ent_Cd": "entity_type",
    "Rndrng_Prvdr_Type": "provider_type",
    "Rndrng_Prvdr_City": "city",
    "Rndrng_Prvdr_State_Abrvtn": "state",
    "Rndrng_Prvdr_Zip5": "zip_code",
    "Rndrng_Prvdr_RUCA": "ruca_code",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "medicare_participating",
    "Tot_HCPCS_Cds": "total_hcpcs_codes",
    "Tot_Benes": "total_beneficiaries",
    "Tot_Srvcs": "total_services",
    "Tot_Sbmtd_Chrg": "total_submitted_charges",
    "Tot_Mdcr_Alowd_Amt": "total_allowed_amount",
    "Tot_Mdcr_Pymt_Amt": "total_medicare_payment",
    "Tot_Mdcr_Stdzd_Amt": "total_standardized_payment",
    "Bene_Avg_Age": "bene_avg_age",
    "Bene_Avg_Risk_Scre": "bene_avg_risk_score",
}
NUMERIC_COLUMNS = [
    "total_hcpcs_codes", "total_beneficiaries", "total_services",
    "total_submitted_charges", "total_allowed_amount", "total_medicare_payment",
    "total_standardized_payment", "bene_avg_age", "bene_avg_risk_score",
]
# Fields without which a provider cannot be compared
REQUIRED_FIELDS = [
    "provider_type", "total_beneficiaries", "total_services",
    "total_submitted_charges", "total_allowed_amount",
    "total_standardized_payment", "bene_avg_risk_score",
]


@dataclass(frozen=True)
class Metric:
    key: str            # column prefix
    label: str          # plain-English name used in reasons
    risk_adjusted: bool


METRICS = [
    Metric("cost", "risk-adjusted Medicare cost per beneficiary", True),
    Metric("services", "risk-adjusted services per beneficiary", True),
    Metric("charge", "submitted-charge to allowed-amount ratio", False),
]
RAW_METRIC = {
    "cost": "cost_per_beneficiary",
    "services": "services_per_beneficiary",
    "charge": "charge_to_allowed_ratio",
}


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename CMS columns, coerce numbers and compute raw metrics.

    Missing or non-positive inputs give a missing metric (never 0), so a
    provider with bad data is reported as unscorable instead of looking cheap.
    """
    missing = [c for c in CMS_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing CMS columns: {missing}")

    df = raw[list(CMS_COLUMNS)].rename(columns=CMS_COLUMNS).copy()
    df["npi"] = df["npi"].astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def ratio(num: str, den: str) -> pd.Series:
        n, d = df[num], df[den]
        valid = (n > 0) & (d > 0)
        return (n / d).where(valid)

    df["cost_per_beneficiary"] = ratio("total_standardized_payment", "total_beneficiaries")
    df["services_per_beneficiary"] = ratio("total_services", "total_beneficiaries")
    df["charge_to_allowed_ratio"] = ratio("total_submitted_charges", "total_allowed_amount")
    df.loc[~(df["bene_avg_risk_score"] > 0), "bene_avg_risk_score"] = np.nan

    present = df[REQUIRED_FIELDS].notna().sum(axis=1)
    df["data_completeness"] = (present / len(REQUIRED_FIELDS)).round(3)
    metric_cols = list(RAW_METRIC.values()) + ["bene_avg_risk_score"]
    df["scorable"] = df[metric_cols].notna().all(axis=1) & df["provider_type"].notna()
    return df


def _modified_z(log_values: pd.Series) -> tuple[pd.Series, float, float]:
    """Iglewicz-Hoaglin modified z-score. Returns (z, median, MAD).

    If MAD is 0 (over half the group has the same value) z is undefined and
    returned as NaN, so nobody is flagged on that metric.
    """
    med = float(log_values.median())
    mad = float((log_values - med).abs().median())
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.nan, index=log_values.index), med, mad
    return 0.6745 * (log_values - med) / mad, med, mad


def _score_group(
    g: pd.DataFrame, z_threshold: float, min_ratio: float
) -> tuple[pd.DataFrame, dict]:
    g = g.copy()
    log_risk = np.log(g["bene_avg_risk_score"])
    summary: dict = {"peer_count": len(g)}

    for m in METRICS:
        log_val = np.log(g[RAW_METRIC[m.key]])
        if m.risk_adjusted:
            if float(log_risk.std()) < 1e-6:
                # Every peer has the same risk score: nothing to adjust for
                slope, intercept = 0.0, 0.0
            else:
                slope, intercept = np.polyfit(log_risk, log_val, 1)
            log_val = log_val - (intercept + slope * log_risk)  # log O/E
            summary[f"{m.key}_risk_slope"] = round(float(slope), 4)
        z, med, _ = _modified_z(log_val)
        g[f"{m.key}_z"] = z
        g[f"{m.key}_ratio_to_peer_median"] = np.exp(log_val - med)
        g[f"{m.key}_percentile"] = (log_val.rank(pct=True) * 100).round(1)
        g[f"_{m.key}_log"] = log_val - med
        summary[f"median_{RAW_METRIC[m.key]}"] = round(float(g[RAW_METRIC[m.key]].median()), 4)

    summary["median_risk_score"] = round(float(g["bene_avg_risk_score"].median()), 4)

    features = g[[f"_{m.key}_log" for m in METRICS]].to_numpy()
    forest = IsolationForest(n_estimators=200, random_state=RANDOM_STATE).fit(features)
    unusual = -forest.score_samples(features)  # higher = more unusual
    g["isolation_forest_percentile"] = (pd.Series(unusual, index=g.index).rank(pct=True) * 100).round(1)

    reasons_col, status_col = [], []
    for _, row in g.iterrows():
        reasons = []
        for m in METRICS:
            z = row[f"{m.key}_z"]
            ratio = row[f"{m.key}_ratio_to_peer_median"]
            if pd.notna(z) and z > z_threshold and ratio >= min_ratio:
                reasons.append(
                    f"{m.label[0].upper()}{m.label[1:]} is {ratio:.1f}x the "
                    f"{row['provider_type']} median (modified z = {z:.1f})"
                )
        reasons_col.append("; ".join(reasons) or None)
        status_col.append(STATUS_OUTLIER if reasons else STATUS_TYPICAL)
    g["outlier_reasons"] = reasons_col
    g["billing_pattern_status"] = status_col
    summary["outlier_count"] = int((g["billing_pattern_status"] == STATUS_OUTLIER).sum())

    return g.drop(columns=[f"_{m.key}_log" for m in METRICS]), summary


def score_providers(
    raw: pd.DataFrame,
    min_peers: int = MIN_PEERS,
    z_threshold: float = Z_THRESHOLD,
    min_ratio: float = MIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every provider against its specialty.

    Returns (providers, peer_groups). Providers whose specialty has fewer
    than ``min_peers`` scorable members get ``insufficient_peers``; providers
    missing required data get ``insufficient_data``. Neither gets comparison
    values.
    """
    df = prepare(raw)
    scorable = df[df["scorable"]]
    sizes = scorable.groupby("provider_type").size()
    eligible = sizes[sizes >= min_peers].index

    scored_parts, summaries = [], []
    for ptype in sorted(eligible):
        part, summary = _score_group(
            scorable[scorable["provider_type"] == ptype], z_threshold, min_ratio
        )
        scored_parts.append(part)
        summaries.append({"provider_type": ptype, **summary})

    result_cols = [
        c for m in METRICS for c in
        (f"{m.key}_z", f"{m.key}_ratio_to_peer_median", f"{m.key}_percentile")
    ] + ["isolation_forest_percentile", "outlier_reasons", "billing_pattern_status"]

    out = df.copy()
    for col in result_cols:
        out[col] = None if col in ("outlier_reasons", "billing_pattern_status") else np.nan
    if scored_parts:
        scored = pd.concat(scored_parts)
        out.loc[scored.index, result_cols] = scored[result_cols]
    unscored = out["billing_pattern_status"].isna()
    out.loc[unscored & ~out["scorable"], "billing_pattern_status"] = STATUS_INSUFFICIENT_DATA
    out.loc[unscored & out["scorable"], "billing_pattern_status"] = STATUS_INSUFFICIENT_PEERS
    out["peer_group_size"] = out["provider_type"].map(sizes).fillna(0).astype(int)

    peer_groups = pd.DataFrame(summaries)
    return out, peer_groups
