"""
Load CMS Medicare provider data into PostgreSQL.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from careagent.config import get_settings
from careagent.db.models import Provider, ProviderService
from careagent.db.session import SessionLocal, check_connection, create_tables

settings = get_settings()

CMS_COLUMN_MAP = {
    "Rndrng_NPI":                     "npi",
    "Rndrng_Prvdr_Last_Org_Name":     "last_name_or_org",
    "Rndrng_Prvdr_First_Name":        "first_name",
    "Rndrng_Prvdr_Crdntls":          "credentials",
    "Rndrng_Prvdr_Gndr":             "gender",
    "Rndrng_Prvdr_Ent_Cd":           "entity_type",
    "Rndrng_Prvdr_St1":              "street",
    "Rndrng_Prvdr_City":             "city",
    "Rndrng_Prvdr_State_Abrvtn":     "state",
    "Rndrng_Prvdr_Zip5":             "zip_code",
    "Rndrng_Prvdr_Cntry":            "country",
    "Rndrng_Prvdr_Type":             "provider_type",
    "Rndrng_Prvdr_Mdcr_Prtcptn_Ind": "medicare_participation",
    "HCPCS_Cd":                       "hcpcs_code",
    "HCPCS_Desc":                     "hcpcs_description",
    "HCPCS_Drug_Ind":                 "is_drug_indicator",
    "Place_Of_Srvc":                  "place_of_service",
    "Tot_Benes":                      "beneficiary_unique_count",
    "Tot_Srvcs":                      "line_service_count",
    "Tot_Bene_Day_Srvcs":            "beneficiary_day_service_count",
    "Avg_Mdcr_Alowd_Amt":            "avg_medicare_allowed_amt",
    "Avg_Sbmtd_Chrg":                "avg_submitted_charge_amt",
    "Avg_Mdcr_Pymt_Amt":             "avg_medicare_payment_amt",
    "Avg_Mdcr_Stdzd_Amt":            "avg_medicare_standardized_amt",
}


def load_raw_cms(filepath: str, limit: int | None = None) -> pd.DataFrame:
    logger.info(f"Reading CMS file: {filepath}")
    df = pd.read_csv(filepath, dtype=str, low_memory=False, nrows=limit)
    logger.info(f"Raw rows loaded: {len(df):,}")
    df = df.rename(columns=CMS_COLUMN_MAP)

    numeric_cols = [
        "line_service_count", "beneficiary_unique_count",
        "beneficiary_day_service_count", "avg_medicare_allowed_amt",
        "avg_submitted_charge_amt", "avg_medicare_payment_amt",
        "avg_medicare_standardized_amt",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "medicare_participation" in df.columns:
        df["medicare_participation"] = df["medicare_participation"].str.upper() == "Y"
    if "is_drug_indicator" in df.columns:
        df["is_drug_indicator"] = df["is_drug_indicator"].str.upper() == "Y"

    df["npi"] = df["npi"].astype(str).str.strip().str.zfill(10)
    logger.info(f"Unique providers: {df['npi'].nunique():,}")
    return df


def build_provider_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating to provider level...")
    identity_cols = [
        "npi", "last_name_or_org", "first_name", "credentials",
        "gender", "entity_type", "street", "city", "state",
        "zip_code", "country", "provider_type", "medicare_participation",
    ]
    identity_cols = [c for c in identity_cols if c in df.columns]
    identity = df.groupby("npi")[identity_cols].first().reset_index(drop=True).assign(npi=df.groupby("npi").groups.keys())

    agg = df.groupby("npi").agg(
        total_services=("line_service_count", "sum"),
        total_unique_beneficiaries=("beneficiary_unique_count", "sum"),
        distinct_procedure_count=("hcpcs_code", "nunique"),
        avg_medicare_payment=("avg_medicare_payment_amt", "mean"),
        avg_submitted_charge=("avg_submitted_charge_amt", "mean"),
        avg_allowed_amount=("avg_medicare_allowed_amt", "mean"),
    ).reset_index()

    providers = identity.merge(agg, on="npi", how="left")
    providers["cms_data_year"] = 2022
    logger.info(f"Provider rows: {len(providers):,}")
    return providers


def upsert_providers(db, providers_df: pd.DataFrame, batch_size: int = 500) -> int:
    records = providers_df.replace({np.nan: None}).to_dict(orient="records")
    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        stmt = insert(Provider).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["npi"],
            set_={
                "last_name_or_org":           stmt.excluded.last_name_or_org,
                "provider_type":              stmt.excluded.provider_type,
                "state":                      stmt.excluded.state,
                "total_services":             stmt.excluded.total_services,
                "total_unique_beneficiaries": stmt.excluded.total_unique_beneficiaries,
                "avg_medicare_payment":       stmt.excluded.avg_medicare_payment,
                "avg_submitted_charge":       stmt.excluded.avg_submitted_charge,
                "avg_allowed_amount":         stmt.excluded.avg_allowed_amount,
                "distinct_procedure_count":   stmt.excluded.distinct_procedure_count,
                "cms_data_year":              stmt.excluded.cms_data_year,
            },
        )
        db.execute(stmt)
        inserted += len(batch)
        if i % 2000 == 0 and i > 0:
            logger.info(f"  Providers upserted: {inserted:,}/{len(records):,}")
    return inserted


def upsert_services(db, services_df: pd.DataFrame, batch_size: int = 1000) -> int:
    service_cols = [
        "npi", "hcpcs_code", "hcpcs_description", "place_of_service",
        "is_drug_indicator", "line_service_count", "beneficiary_unique_count",
        "beneficiary_day_service_count", "avg_medicare_allowed_amt",
        "avg_submitted_charge_amt", "avg_medicare_payment_amt",
        "avg_medicare_standardized_amt",
    ]
    service_cols = [c for c in service_cols if c in services_df.columns]
    df = services_df[service_cols].copy()
    df["cms_data_year"] = 2022
    records = df.replace({np.nan: None}).to_dict(orient="records")

    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        db.execute(insert(ProviderService).values(batch).on_conflict_do_nothing())
        inserted += len(batch)
        if i % 10000 == 0 and i > 0:
            logger.info(f"  Services inserted: {inserted:,}/{len(records):,}")
    return inserted


def main(filepath: str, limit: int | None = None) -> None:
    start = time.perf_counter()
    logger.info("CareAgent — CMS Data Loader")

    if not check_connection():
        logger.error("Cannot connect to PostgreSQL. Is docker-compose up?")
        sys.exit(1)

    create_tables()
    df = load_raw_cms(filepath, limit=limit)
    providers_df = build_provider_aggregates(df)

    with SessionLocal() as db:
        logger.info("Upserting providers...")
        n_providers = upsert_providers(db, providers_df)
        logger.info(f"Providers upserted: {n_providers:,}")

        logger.info("Inserting services...")
        n_services = upsert_services(db, df)
        logger.info(f"Services inserted: {n_services:,}")
        db.commit()

    elapsed = time.perf_counter() - start
    logger.info(f"Done in {elapsed:.1f}s — {n_providers:,} providers, {n_services:,} services")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=settings.cms_data_path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(filepath=args.file, limit=args.limit)
