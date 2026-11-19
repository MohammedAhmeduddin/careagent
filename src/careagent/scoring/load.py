"""
careagent.scoring.load
~~~~~~~~~~~~~~~~~~~~~~
Turn scored DataFrames into database rows and load them.
"""

from __future__ import annotations

import math

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from careagent.db.models import PeerGroup, Provider
from careagent.db.queries import replace_all_data
from careagent.scoring.peer import score_providers

_PROVIDER_COLUMNS = [c.name for c in Provider.__table__.columns
                     if c.name not in ("created_at",)]
_PEER_COLUMNS = [c.name for c in PeerGroup.__table__.columns]


def _clean(value):
    """NaN/NA -> None, numpy scalars -> Python scalars."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
    return value


def provider_rows(scored: pd.DataFrame, year: int) -> list[dict]:
    df = scored.assign(cms_data_year=year)
    cols = [c for c in _PROVIDER_COLUMNS if c in df.columns]
    rows = []
    for rec in df[cols].to_dict("records"):
        row = {k: _clean(v) for k, v in rec.items()}
        if row.get("peer_group_size") is not None:
            row["peer_group_size"] = int(row["peer_group_size"])
        rows.append(row)
    return rows


def peer_group_rows(groups: pd.DataFrame, year: int) -> list[dict]:
    if groups.empty:
        return []
    df = groups.assign(cms_data_year=year)
    return [{k: _clean(v) for k, v in rec.items()}
            for rec in df[_PEER_COLUMNS].to_dict("records")]


def load_dataframe(db: Session, raw: pd.DataFrame, year: int) -> dict:
    """Score ``raw`` CMS rows and replace the database contents. Returns counts."""
    scored, groups = score_providers(raw)
    providers = provider_rows(scored, year)
    peers = peer_group_rows(groups, year)
    replace_all_data(db, providers, peers)
    counts = scored["billing_pattern_status"].value_counts().to_dict()
    logger.info(f"Loaded {len(providers)} providers, {len(peers)} peer groups: {counts}")
    return {"providers": len(providers), "peer_groups": len(peers), **counts}
