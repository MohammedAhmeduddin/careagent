"""
Download real CMS "Medicare Physician & Other Practitioners - by Provider" rows
for one state from the data.cms.gov API and save them as CSV.

One row per provider (NPI) with totals for services, beneficiaries, submitted
charges, Medicare allowed and paid amounts, and beneficiary characteristics.

Usage:
    python scripts/download_cms_providers.py --state NJ
    python scripts/download_cms_providers.py --state NJ --limit 2000   # quick test

Default dataset: data year 2024 (release "by Provider : 2024-12-01").
"""

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_2024 = "4d0b2df0-1e99-4db7-a574-a571d99217f1"
API = "https://data.cms.gov/data-api/v1/dataset/{dataset}/data"
PAGE_SIZE = 5000


def fetch_page(dataset: str, state: str, offset: int, size: int, retries: int = 3) -> list[dict]:
    params = {"size": size, "offset": offset, "filter[Rndrng_Prvdr_State_Abrvtn]": state}
    url = API.format(dataset=dataset) + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception as e:  # network hiccups: retry with backoff
            if attempt == retries:
                raise
            print(f"  retry {attempt} after error: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    return []


def download(dataset: str, state: str, limit: int | None, out: Path, fetch=fetch_page) -> int:
    rows, offset = [], 0
    while True:
        size = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - len(rows))
        if size <= 0:
            break
        page = fetch(dataset, state, offset, size)
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        print(f"  fetched {len(rows):,} rows")
        if len(page) < size:
            break
    if not rows:
        raise SystemExit(f"No rows returned for state={state}; check the dataset id")

    columns = list(rows[0].keys())
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows):,} providers x {len(columns)} columns to {out}")
    print("Columns:", ", ".join(columns))
    return len(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="NJ")
    parser.add_argument("--dataset", default=DATASET_2024)
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many rows")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or Path(f"data/raw/cms_providers_{args.state}_2024.csv")
    download(args.dataset, args.state.upper(), args.limit, out)


if __name__ == "__main__":
    main()
