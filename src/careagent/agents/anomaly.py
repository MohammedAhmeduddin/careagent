"""
careagent.agents.anomaly
~~~~~~~~~~~~~~~~~~~~~~~~~
Anomaly Detection Agent — Isolation Forest on cost + quality metrics.

How it works:
1. Loads all providers with sufficient volume from the database
2. Fits Isolation Forest on 4 features:
   - avg_submitted_charge
   - avg_medicare_payment
   - cost_efficiency_ratio (from Statistical Agent)
   - total_services
3. Scores the target provider
4. Flags if in top contamination% (default 3%)

Why Isolation Forest:
- Works well on tabular data with no labeled anomalies
- Interpretable contamination parameter
- Fast inference on 10K providers
- Industry standard for unsupervised outlier detection
"""

import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from careagent.graph.state import AgentState
from careagent.db.session import get_db
from careagent.db.queries import (
    get_provider_by_npi,
    get_all_providers_for_anomaly_detection,
    update_provider_anomaly,
)
from careagent.config import get_settings

settings = get_settings()

# Features used for anomaly detection
ANOMALY_FEATURES = [
    "avg_submitted_charge",
    "avg_medicare_payment",
    "avg_allowed_amount",
    "total_services",
]


def _build_feature_matrix(providers) -> tuple[np.ndarray, list[str]]:
    """
    Build feature matrix from provider list.
    Returns (X, npi_list) where X is shape (n_providers, n_features).
    """
    rows = []
    npis = []
    for p in providers:
        row = [
            p.avg_submitted_charge   or 0.0,
            p.avg_medicare_payment   or 0.0,
            p.avg_allowed_amount     or 0.0,
            p.total_services         or 0.0,
        ]
        rows.append(row)
        npis.append(p.npi)
    return np.array(rows, dtype=float), npis


def _explain_anomaly(provider, score: float, is_anomaly: bool) -> str:
    """
    Generate a plain-English explanation of the anomaly result.
    """
    if not is_anomaly:
        return (
            f"Provider cost and volume metrics are within normal range for "
            f"{provider.provider_type}. "
            f"Anomaly score: {score:.4f}."
        )

    reasons = []
    if provider.avg_submitted_charge and provider.avg_medicare_payment:
        ratio = provider.avg_submitted_charge / max(provider.avg_medicare_payment, 1)
        if ratio > 3.0:
            reasons.append(
                f"submitted charges are {ratio:.1f}x higher than Medicare payments"
            )
    if provider.total_services and provider.total_services > 5000:
        reasons.append(f"unusually high service volume ({provider.total_services:.0f})")
    if provider.avg_submitted_charge and provider.avg_submitted_charge > 1000:
        reasons.append(
            f"high average submitted charge (${provider.avg_submitted_charge:.2f})"
        )

    reason_text = "; ".join(reasons) if reasons else "cost/volume pattern deviates from specialty peers"
    return (
        f"Flagged as anomaly: {reason_text}. "
        f"Isolation Forest score: {score:.4f} "
        f"(more negative = more anomalous)."
    )


def anomaly_agent(state: AgentState) -> AgentState:
    """
    Runs Isolation Forest anomaly detection on the target provider.

    Steps:
    1. Load all scoreable providers (training set)
    2. Build + fit Isolation Forest
    3. Score the target provider
    4. Write flag, score, and explanation to state + database
    """
    npi = state["npi"]
    logger.info(f"[Anomaly] Processing NPI={npi}")

    with get_db() as db:
        # ── Load training data ─────────────────────────────────────────────────
        all_providers = get_all_providers_for_anomaly_detection(
            db, min_services=10.0
        )

        if len(all_providers) < 10:
            logger.warning(f"[Anomaly] Too few providers ({len(all_providers)}) — skipping")
            executed = state.get("agents_executed", []) + ["anomaly"]
            return {
                **state,
                "is_anomaly":       False,
                "anomaly_score":    0.0,
                "anomaly_reason":   "Insufficient data for anomaly detection",
                "anomaly_complete": True,
                "agents_executed":  executed,
            }

        # ── Build feature matrix ───────────────────────────────────────────────
        X, npi_list = _build_feature_matrix(all_providers)

        # ── Scale features ─────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Fit Isolation Forest ───────────────────────────────────────────────
        iso_forest = IsolationForest(
            contamination=settings.anomaly_contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        iso_forest.fit(X_scaled)
        logger.info(
            f"[Anomaly] Isolation Forest fitted on {len(all_providers)} providers "
            f"(contamination={settings.anomaly_contamination})"
        )

        # ── Score target provider ──────────────────────────────────────────────
        target = get_provider_by_npi(db, npi)
        if not target:
            return {**state, "error": f"Provider {npi} not found"}

        target_features = np.array([[
            target.avg_submitted_charge   or 0.0,
            target.avg_medicare_payment   or 0.0,
            target.avg_allowed_amount     or 0.0,
            target.total_services         or 0.0,
        ]])
        target_scaled  = scaler.transform(target_features)
        raw_score      = float(iso_forest.score_samples(target_scaled)[0])
        prediction     = iso_forest.predict(target_scaled)[0]  # -1=anomaly, 1=normal
        is_anomaly     = prediction == -1
        anomaly_score  = round(raw_score, 6)

        # ── Compute anomaly rate for logging ───────────────────────────────────
        all_predictions = iso_forest.predict(X_scaled)
        anomaly_count   = int((all_predictions == -1).sum())
        anomaly_rate    = round(anomaly_count / len(all_providers) * 100, 2)
        logger.info(
            f"[Anomaly] Flagged {anomaly_count}/{len(all_providers)} "
            f"providers ({anomaly_rate}%)"
        )

        # ── Generate explanation ───────────────────────────────────────────────
        reason = _explain_anomaly(target, anomaly_score, is_anomaly)

        # ── Write to database ──────────────────────────────────────────────────
        update_provider_anomaly(
            db, npi,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_reason=reason,
        )

    executed = state.get("agents_executed", []) + ["anomaly"]
    logger.info(f"[Anomaly] NPI={npi} is_anomaly={is_anomaly} score={anomaly_score:.4f}")

    return {
        **state,
        "is_anomaly":       is_anomaly,
        "anomaly_score":    anomaly_score,
        "anomaly_reason":   reason,
        "anomaly_complete": True,
        "agents_executed":  executed,
    }
