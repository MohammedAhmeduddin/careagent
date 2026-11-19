<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangGraph-Multi--Agent-7C3AED?style=flat-square"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-113%20passing-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/Coverage-98%25%20(whole%20package)-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white"/>
<img src="https://img.shields.io/badge/GCP-Cloud%20Run-4285F4?style=flat-square&logo=googlecloud&logoColor=white"/>

<br/><br/>

# CareAgent

### Multi-agent analysis of Medicare billing patterns on real CMS data

*Compares each of **39,694 New Jersey Medicare providers** with peers in the same specialty, after adjusting for how sick their patients are. Unusual billing patterns are flagged with reasons computed from the data, and a LangGraph pipeline of five agents turns each comparison into a scorecard in about 26 ms.*

<br/>

[![Live Dashboard](https://img.shields.io/badge/Dashboard-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/AhmeduddinMohammed/careagent)
[![Live API](https://img.shields.io/badge/API-GCP%20Cloud%20Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://careagent-api-668260909878.us-central1.run.app)
[![API Docs](https://img.shields.io/badge/Docs-Swagger-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://careagent-api-668260909878.us-central1.run.app/docs)

</div>

> **What this is and is not.** CareAgent compares Medicare *billing patterns* with same-specialty peers. Billing data contains no clinical outcomes, so this is **not a measure of quality of care**, and an outlier is **not evidence of fraud or wrongdoing**. Every scorecard carries this disclaimer. Provider names and street addresses are deliberately not stored.

---

## Highlights & Impact

| | |
|---|---|
| 🏥 **Real data, full state** | **39,694** NJ providers × 81 columns from the CMS *Medicare Physician & Other Practitioners – by Provider* 2024 release, pulled from the CMS API by a paginated, retrying downloader |
| 👥 **Peer groups** | **71** specialties with ≥30 providers (31 to 4,852 peers, median 261); **39,471** providers compared; 223 in smaller specialties are explicitly *not* compared instead of being forced into a bad benchmark |
| ⚖️ **Risk-adjusted** | Cost and utilization adjusted per specialty for patient risk (CMS HCC score) via a log-log regression fitted within each specialty. Slopes range from −1.4 to +2.9 across specialties, which is why one global adjustment would be wrong |
| 🚩 **Explainable outliers** | **1,461** outliers (**3.7%**) under a two-part rule: modified z > 3.5 (Iglewicz–Hoaglin) **and** ≥ 2× the specialty median. Every flag lists the exact metrics and ratios that triggered it |
| 🌲 **Independent check** | A per-specialty Isolation Forest ranks flagged providers at a median **96.4th** percentile of unusualness vs **48.2** for typical providers. Two different methods agree |
| ⚡ **Fast** | Scoring runs once at load (39,694 providers scored and written in ~30 s); the API reads stored results, giving **p50 26 ms / p95 45 ms** for the full 5-agent pipeline (200 random real providers, local Postgres) |
| 🧠 **Grounded LLM** | The narrative model (gpt-6-luna) must use only numbers it was given: a number check rejects any narrative containing an unsupported figure and falls back to a deterministic template |
| ✅ **Tested** | **113 tests** (101 unit + 12 PostgreSQL integration), **98.6%** coverage of the *whole* package; CI runs the integration tests against a Postgres service container |

---

## How it works

```
CMS API ──► download_cms_providers.py ──► CSV (39,694 × 81)
                                            │
                       load_cms_providers.py│  score once, per specialty:
                                            │  metrics → risk adjustment → robust z
                                            │  → outlier rule → Isolation Forest rank
                                            ▼
                          PostgreSQL: providers · peer_groups · agent_runs
                                            │
POST /analyze {npi} ──► LangGraph supervisor ──► 5 agents ──► scorecard (JSON)
```

### The comparison (`src/careagent/scoring/peer.py`)

| Metric | Formula | Risk-adjusted |
|---|---|---|
| Cost per beneficiary | standardized Medicare payment ÷ beneficiaries (CMS standardization removes geographic pay differences) | ✅ |
| Services per beneficiary | services ÷ beneficiaries | ✅ |
| Charge-to-allowed ratio | submitted charges ÷ Medicare allowed amount | ❌ (a pricing choice, not driven by patient risk) |

1. **Risk adjustment.** Within each specialty, fit `log(metric) ~ log(avg HCC risk score)`. The observed ÷ expected ratio is what's left after patient risk. A provider with sicker patients isn't penalized for higher costs (a unit test checks exactly this).
2. **Robust z-score.** Modified z = 0.6745 × (x − median) / MAD on the log O/E ratio. It uses median and MAD rather than mean and SD, so the outliers being hunted don't distort the yardstick. If MAD = 0, nobody is flagged on that metric.
3. **Outlier rule.** Modified z > 3.5 **and** ratio ≥ 2× the specialty median, high side only. The second condition stops tight specialties from flagging trivial differences. Without it, 39% of one specialty (Opioid Treatment Programs) was flagged.
4. **Isolation Forest.** Fitted per specialty on the three log ratios. It *ranks* overall unusualness and does not decide the flag, so every flag stays explainable.
5. **Missing data stays missing.** No zero-filling and no imputation. A provider with a missing or invalid required field is reported as `insufficient_data`.

### The agents (`src/careagent/agents/`)

A LangGraph `StateGraph`: the supervisor reads the shared state after every step and decides what runs next.

| Agent | Does |
|---|---|
| 🧹 **Data Validator** | Checks the 7 required fields are present and positive and lists any that aren't. Never imputes: an imputed median would make a provider look average |
| 📊 **Statistical** | Loads the provider's peer comparison and specialty benchmarks (medians, peer count) |
| 🔍 **Anomaly** | Outlier flag, reasons, Isolation Forest percentile. **Skipped by the supervisor** when the specialty has too few peers or data is missing, because there is nothing to compare |
| ✍️ **Summarizer** | 2–3 sentence narrative from the configured OpenAI model via tool calling, number-checked; template fallback on no key, error or failed check |
| 📋 **Reporter** | Final scorecard: status (`typical` / `outlier` / `insufficient_peers` / `insufficient_data`) + disclaimer |

Two providers can take different paths: an outlier runs all five agents, while a provider in a 10-person specialty skips Anomaly and its scorecard explains why.

---

## Example (`POST /analyze`)

```json
{
  "npi": "1003049610",
  "provider_type": "Critical Care (Intensivists)",
  "city": "Pennington", "state": "NJ",
  "billing_pattern_status": "outlier",
  "peer_group_size": 140,
  "peer_comparison": {
    "cost":     {"ratio_to_peer_median": 1.069, "percentile": 60.7, "modified_z": 0.17},
    "services": {"ratio_to_peer_median": 2.45,  "percentile": 96.4, "modified_z": 3.87},
    "charge":   {"...": "..."}
  },
  "outlier_reasons": [
    "Risk-adjusted services per beneficiary is 2.5x the Critical Care (Intensivists) median (modified z = 3.9)"
  ],
  "agents_executed": ["data_cleaner", "statistical", "anomaly", "summarizer", "reporter"],
  "agents_skipped": [],
  "disclaimer": "Compares Medicare billing patterns with same-specialty peers after adjusting for patient risk. It is not a measure of quality of care, and an outlier is not evidence of fraud or wrongdoing."
}
```

`GET /` lists one real example NPI per status; `GET /health` reports providers loaded, compared and flagged, and the number of peer groups.

| Code | When |
|---|---|
| 200 | Scorecard |
| 404 | NPI not in the dataset |
| 422 | NPI not exactly 10 digits |
| 500 | Pipeline failure. Returns a run ID only: details go to logs and `agent_runs`, never to the caller |

---

## Quickstart

```bash
git clone https://github.com/MohammedAhmeduddin/careagent && cd careagent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"
cp .env.example .env            # local Docker Postgres: POSTGRES_PORT=5433, POSTGRES_SSLMODE=disable

docker compose up -d postgres
python scripts/download_cms_providers.py --state NJ                       # ~15 MB from data.cms.gov
python scripts/load_cms_providers.py --csv data/raw/cms_providers_NJ_2024.csv

PYTHONPATH=src:. pytest                                                   # unit tests
RUN_INTEGRATION=1 PYTHONPATH=src:. pytest                                 # + integration (LOCAL db only: replaces data)

uvicorn careagent.api.main:app --app-dir src --port 8080
CAREAGENT_API=http://localhost:8080 streamlit run app.py
```

Any state works: `--state CA` downloads California. Upgrading a database created by v1 needs `--reset-schema` once.

---

## Engineering notes

- **Score once, read many.** Percentiles and outlier thresholds only mean something against the whole peer group, so they are computed in one batch at load time. Each request is a few primary-key reads, with no model refit per request.
- **Config.** `DATABASE_URL` (one secret, e.g. from GCP Secret Manager) or `POSTGRES_*` + `POSTGRES_SSLMODE`. `postgres://` URLs are normalized to the psycopg2 driver.
- **Safety rails.** Integration tests run only with `RUN_INTEGRATION=1` *and* a localhost database, so a test run can never wipe the production database. API errors never echo exception text.
- **CI.** GitHub Actions on every push and PR: all tests against Postgres 16 with an 85% whole-package coverage gate, plus a blocking `ruff` lint (version pinned).

## Project structure

```
src/careagent/
├── scoring/peer.py      # metrics, risk adjustment, robust z, outlier rule, Isolation Forest
├── scoring/load.py      # scored DataFrame → database rows
├── agents/              # supervisor + 5 agents
├── graph/               # AgentState contract + LangGraph wiring
├── db/                  # SQLAlchemy models (providers, peer_groups, agent_runs), queries, session
├── api/                 # FastAPI app, routes, Pydantic schemas
└── config.py
scripts/download_cms_providers.py   # CMS API → CSV (paginated, retries)
scripts/load_cms_providers.py       # CSV → score → Postgres
app.py                              # Streamlit dashboard
tests/unit · tests/integration      # 101 + 12 tests
```

## Limitations

- **Billing, not quality.** No outcomes, readmissions or patient experience are in the data.
- **Medicare fee-for-service only.** Medicare Advantage and commercial patients are invisible; a provider with few traditional Medicare patients may look unusual for that reason alone.
- **Specialty is self-reported** and broad. Sub-specialists (e.g. interventional vs general cardiology) can differ legitimately inside one group.
- **Services are counted as CMS reports them.** Anesthesia, for example, is billed in time units, which is why comparisons are only ever made within a specialty.
- **CMS suppresses small cells** (<11 beneficiaries), so very low-volume providers are absent.

## Data

[CMS Medicare Physician & Other Practitioners – by Provider](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider), 2024 data year, New Jersey. Public data. The raw CSV is not committed (`data/` is gitignored); the downloader recreates it.

---

## Author

**Mohammed Ahmeduddin**: [GitHub](https://github.com/MohammedAhmeduddin) · [LinkedIn](https://www.linkedin.com/in/mohammed-ahmeduddin/) · Other projects: [SessionScout](https://huggingface.co/spaces/AhmeduddinMohammed/sessionscout) · [LoanLens](https://huggingface.co/spaces/AhmeduddinMohammed/loanlens)
