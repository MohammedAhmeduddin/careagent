"""
Load CMS "by Provider" data, score every provider against its specialty
peers and replace the providers / peer_groups tables.

    python scripts/download_cms_providers.py --state NJ
    python scripts/load_cms_providers.py --csv data/raw/cms_providers_NJ_2024.csv

Uses the database configured in .env (DATABASE_URL or POSTGRES_*).
Existing providers, peer groups and agent runs are replaced. Pass
--reset-schema once when upgrading a database created by the old schema.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from careagent.db.session import create_tables, get_db, reset_schema  # noqa: E402
from careagent.scoring.load import load_dataframe  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", default="data/raw/cms_providers_NJ_2024.csv")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--reset-schema", action="store_true",
        help="Drop and recreate all tables first (needed once when moving from the old schema)",
    )
    args = parser.parse_args(argv)

    raw = pd.read_csv(args.csv, dtype=str)
    print(f"Read {len(raw):,} rows from {args.csv}")
    if args.reset_schema:
        reset_schema()
    else:
        create_tables()
    with get_db() as db:
        counts = load_dataframe(db, raw, year=args.year)
    for key, value in counts.items():
        print(f"  {key:20s} {value:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
