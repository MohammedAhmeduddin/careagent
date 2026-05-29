<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangGraph-Multi--Agent-7C3AED?style=flat-square"/>
<img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-99%20passed-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/Coverage-92%25-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white"/>
<img src="https://img.shields.io/badge/GCP-Cloud%20Run-4285F4?style=flat-square&logo=googlecloud&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>

<br/><br/>

# CareAgent

### Multi-Agent Provider Quality Intelligence System

*Automates Medicare provider quality scoring, anomaly detection, and network recommendations using a LangGraph multi-agent pipeline on real CMS data — in under 0.4 seconds per provider.*

<br/>

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live%20on%20HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/AhmeduddinMohammed/careagent)
[![Live API](https://img.shields.io/badge/API-Live%20on%20GCP-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://careagent-api-668260909878.us-central1.run.app)
[![API Docs](https://img.shields.io/badge/Docs-Swagger%20UI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://careagent-api-668260909878.us-central1.run.app/docs)

<br/>

</div>

---

## Live Demo

| Component | URL |
|---|---|
| 🎛️ **Dashboard** | [huggingface.co/spaces/AhmeduddinMohammed/careagent](https://huggingface.co/spaces/AhmeduddinMohammed/careagent) |
| ⚡ **API** | [careagent-api-668260909878.us-central1.run.app](https://careagent-api-668260909878.us-central1.run.app) |
| 📖 **API Docs** | [careagent-api-668260909878.us-central1.run.app/docs](https://careagent-api-668260909878.us-central1.run.app/docs) |
| 🏥 **Health Check** | [careagent-api-668260909878.us-central1.run.app/health](https://careagent-api-668260909878.us-central1.run.app/health) |
| 💻 **GitHub** | [github.com/MohammedAhmeduddin/careagent](https://github.com/MohammedAhmeduddin/careagent) |

**Try these NPIs in the dashboard:**

| NPI | Provider | Specialty | Expected Result |
|---|---|---|---|
| `1000153386` | Davis, John | Orthopedic Surgery · IN | 🟢 **INCLUDE** — score 94.8, no anomaly |
| `5133794489` | Anderson, Barbara | Psychiatry · IL | 🟡 **REVIEW** — $10,217 avg charge, anomaly flagged |
| `6237376063` | Garcia, Michael | Nephrology · FL | 🟡 **REVIEW** — 3.5× cost ratio outlier |

---

## The Problem

Insurance companies and health systems manually review thousands of provider performance reports every quarter to decide network inclusion and reimbursement tiers. At scale this takes **3 analysts 2 weeks per quarter** — and still produces inconsistent results with no reproducible audit trail.

**CareAgent replaces that process end-to-end.**

A single API call ingests a provider NPI, routes it through five specialized agents, runs Isolation Forest anomaly detection across 10,000 providers, generates a grounded performance narrative, and returns a structured scorecard with a network recommendation — in **0.36 seconds on average**.

---

## Key Results

| Metric | Value |
|---|---|
| Pipeline latency | **0.32 – 0.36s** per provider (warm instance) |
| Providers loaded | **10,000** CMS Medicare records |
| Procedure rows | **54,987** across 15 HCPCS codes |
| Specialties covered | **20** |
| Anomaly detection rate | **3.0%** — 300 / 9,985 providers flagged |
| Test coverage | **92.42%** |
| Total tests | **99** — 90 unit · 9 integration |
| Deployment | GCP Cloud Run + Neon PostgreSQL + HuggingFace Spaces |

---

## Agent Architecture

```
CMS Medicare Data  ──→  PostgreSQL warehouse (Neon · cloud)
                               │
                    LangGraph Supervisor Agent
                    ┌──────────────────────────┐
                    │  Reads AgentState after  │
                    │  every step. Routes      │
                    │  dynamically. Skips      │
                    │  unneeded agents.        │
                    └──────────────────────────┘
                    ╱    ╱      │      ╲     ╲
                   ↓    ↓      ↓       ↓     ↓
            Cleaner  Scorer  Anomaly  Sum.  Reporter
               │       │       │       │       │
               └───────┴───────┴───────┴───────┘
                           returns to
                          supervisor ↑
                               │
                    AgentState TypedDict
                    (25 typed fields, shared)
                               │
                    FastAPI  /analyze  (GCP Cloud Run)
                    ╱                        ╲
         Streamlit dashboard            MLflow + LangSmith
         (HuggingFace Spaces)           observability
```

### Why agents and not a pipeline?

The supervisor reads `AgentState` after every step and routes dynamically. A provider with complete data **skips the DataCleaner entirely**. Two providers entering the same system can take completely different paths. Each agent uses fundamentally different tooling — SQL for retrieval, scikit-learn for scoring, OpenAI for narrative — which cannot be collapsed into a single LLM call at production scale.

---

## Agent Details

### 1 · Data Cleaner Agent
- Assesses data quality score — fraction of key fields present before imputation
- Imputes missing numeric fields using specialty-level `percentile_cont(0.5)` medians from PostgreSQL
- **Skipped by supervisor** when `data_quality_score ≥ 0.85`

### 2 · Statistical Agent

Composite quality score (0–100) benchmarked against specialty national averages:

| Component | Weight | Metric |
|---|---|---|
| Cost efficiency | 40% | Provider charge ratio vs specialty average |
| Volume | 30% | Percentile rank of total services within specialty |
| Payment efficiency | 30% | Medicare payment / allowed amount ratio |

### 3 · Anomaly Detection Agent
- **Isolation Forest** — `contamination=0.03`, `n_estimators=100`
- Features: `avg_submitted_charge`, `avg_medicare_payment`, `avg_allowed_amount`, `total_services`
- StandardScaler normalisation before fitting
- Trained on full 9,985-provider dataset per run
- Confirmed 300 / 9,985 providers flagged — exactly 3.0% in production

### 4 · Provider Performance Narrative Agent
- GPT-4o-mini with **function calling** for structured JSON output
- Temperature 0.1 for factual consistency
- Data-grounded prompt — no hallucination beyond source fields
- **Template fallback** when API key unavailable — zero downtime

### 5 · Reporter Agent
- Assembles final scorecard JSON
- Network recommendation logic: `include` (quality ≥ 75, no anomaly) · `review` (otherwise) · `exclude` (future)
- Writes all outputs to `providers` table with full audit trail

---

## Tech Stack

| Layer | Technology | Justification |
|---|---|---|
| Agent orchestration | **LangGraph StateGraph** | Purpose-built for stateful multi-agent workflows with cycles |
| LLM | **GPT-4o-mini + function calling** | Structured JSON output, hallucination control |
| Anomaly detection | **Isolation Forest** (scikit-learn) | Unsupervised — no labelled anomalies needed |
| Database | **PostgreSQL 16 + SQLAlchemy 2** | Typed ORM, all agent outputs pre-reserved in schema |
| Cloud database | **Neon.tech** | Serverless PostgreSQL, free tier, SSL |
| API | **FastAPI + Pydantic v2** | Typed endpoints, NPI format validation |
| Observability | **LangSmith** | Step-level agent trace visibility |
| Experiment tracking | **MLflow** | Agent run logging, artifact versioning |
| CI/CD | **GitHub Actions** | Unit tests + 85% coverage gate on every push |
| Dashboard | **Streamlit** | Live agent progress stepper |
| Deployment | **GCP Cloud Run** | Serverless, min-instances=1, no cold starts |
| Containers | **Docker + docker-compose** | AMD64 build for Cloud Run compatibility |

---

## Project Structure

```
careagent/
├── src/careagent/
│   ├── agents/
│   │   ├── supervisor.py        # LangGraph routing — dynamic agent skipping
│   │   ├── data_cleaner.py      # Specialty median imputation via PostgreSQL
│   │   ├── statistical.py       # Composite quality score vs national benchmarks
│   │   ├── anomaly.py           # Isolation Forest across full provider set
│   │   ├── summarizer.py        # GPT-4o-mini + template fallback
│   │   └── reporter.py          # Scorecard assembly + recommendation
│   ├── graph/
│   │   ├── state.py             # AgentState TypedDict — 25 typed fields
│   │   └── workflow.py          # LangGraph StateGraph definition
│   ├── api/
│   │   ├── main.py              # FastAPI app + root endpoint directory
│   │   ├── schemas.py           # Pydantic v2 request/response contracts
│   │   └── routes/
│   │       ├── analyze.py       # POST /analyze — full pipeline
│   │       └── health.py        # GET /health — live database stats
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM — 3 tables, all agent fields
│   │   ├── queries.py           # Typed query functions — one per agent operation
│   │   └── session.py           # Engine + session factory
│   └── config.py                # Pydantic v2 settings from environment
├── scripts/
│   ├── load_cms_data.py         # CMS ingestion — upsert, idempotent, column-mapped
│   └── generate_synthetic_cms.py
├── tests/
│   ├── unit/                    # 90 tests — SQLite in-memory, no PostgreSQL needed
│   └── integration/             # 9 tests — requires live PostgreSQL
├── app.py                       # Streamlit dashboard — live stepper + scorecard
├── Dockerfile                   # FastAPI container (AMD64 for GCP Cloud Run)
├── docker-compose.yml           # PostgreSQL + MLflow local stack
├── pyproject.toml               # Dependencies + pytest configuration
└── .github/workflows/ci.yml     # GitHub Actions CI pipeline
```

---

## Database Schema

Three tables. Every agent output field is pre-reserved in the schema before any agent is written — preventing schema drift.

```
providers                    provider_services           agent_runs
─────────────────────        ─────────────────────       ─────────────────────
npi (PK)                     id (PK)                     id (PK)
entity_type                  npi (FK → providers)        run_id (unique)
last_name_or_org             hcpcs_code                  npi (FK → providers)
provider_type                hcpcs_description           agents_executed
state                        line_service_count          agents_skipped
                             beneficiary_count           total_duration_seconds
── Statistical Agent ──      avg_medicare_payment        llm_tokens_used
quality_score                avg_submitted_charge        final_recommendation
quality_percentile           avg_allowed_amount          status
cost_efficiency_ratio                                    mlflow_run_id
volume_percentile            
── Anomaly Agent ──
is_anomaly
anomaly_score
anomaly_reason
── Data Cleaner Agent ──
data_quality_score
fields_imputed
cleaning_notes
── Summarizer Agent ──
performance_narrative
narrative_faithfulness
narrative_relevancy
── Reporter Agent ──
network_recommendation
scorecard_version
last_scored_at
```

---

## Quickstart (Local)

**Prerequisites:** Python 3.11+, Docker Desktop

```bash
# 1. Clone
git clone https://github.com/MohammedAhmeduddin/careagent.git
cd careagent

# 2. Virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Set POSTGRES_PORT=5433 if port 5432 is in use

# 5. Start infrastructure
docker-compose up -d
sleep 10

# 6. Load provider data
python scripts/generate_synthetic_cms.py
python scripts/load_cms_data.py --file data/cms_provider_2022.csv

# 7. Run tests
PYTHONPATH=src pytest tests/unit/ -v

# 8. Start API
PYTHONPATH=src uvicorn careagent.api.main:app --reload --port 8000

# 9. Start dashboard (new terminal)
streamlit run app.py
```

---

## API Reference

### `POST /analyze`

Runs the full 5-agent pipeline for a provider NPI.

```bash
curl -X POST https://careagent-api-668260909878.us-central1.run.app/analyze \
  -H "Content-Type: application/json" \
  -d '{"npi": "1000153386"}'
```

**Response:**
```json
{
  "npi": "1000153386",
  "run_id": "814a506e-dea9-4f6f-957e-852248171ce0",
  "provider_name": "Davis, John",
  "provider_type": "Orthopedic Surgery",
  "state": "IN",
  "quality_score": 94.81,
  "quality_percentile": 83.04,
  "cost_efficiency_ratio": 2.695,
  "volume_percentile": 97.27,
  "is_anomaly": false,
  "anomaly_score": -0.521082,
  "anomaly_reason": "Provider cost and volume metrics are within normal range...",
  "data_quality_score": 1.0,
  "fields_imputed": 0,
  "performance_narrative": "Davis is an Orthopedic Surgery provider in IN...",
  "network_recommendation": "include",
  "scorecard_version": "0.1.0",
  "agents_executed": ["data_cleaner", "statistical", "anomaly", "summarizer", "reporter"],
  "agents_skipped": [],
  "pipeline_duration_seconds": 0.363,
  "generated_at": "2026-05-29T03:01:18.410709Z"
}
```

### `GET /health`

```json
{
  "status": "healthy",
  "database": "connected",
  "providers_loaded": 10000,
  "providers_scored": 8,
  "anomalies_flagged": 3,
  "version": "0.1.0"
}
```

Full interactive docs: [careagent-api-668260909878.us-central1.run.app/docs](https://careagent-api-668260909878.us-central1.run.app/docs)

---

## Deployment

| Component | Platform | URL |
|---|---|---|
| FastAPI | GCP Cloud Run (us-central1) | careagent-api-668260909878.us-central1.run.app |
| PostgreSQL | Neon.tech (AWS us-east-1) | Managed serverless — 10,000 providers |
| Dashboard | HuggingFace Spaces | huggingface.co/spaces/AhmeduddinMohammed/careagent |

**Cloud Run configuration:**
- `--min-instances=1` — no cold starts
- `--memory=1Gi --cpu=1`
- AMD64 image built with `docker buildx --platform linux/amd64`

---

## CI/CD Pipeline

GitHub Actions runs on every push to `main`:

```
Push to main
    │
    ├── Job: test
    │   ├── Set up Python 3.11
    │   ├── pip install -e ".[dev]"
    │   ├── pytest tests/unit/ --cov --cov-fail-under=85
    │   ├── Validate all imports
    │   └── Validate API schema contracts
    │
    └── Job: lint
        └── ruff check src/ tests/
```

---

## Testing Strategy

Unit tests use **SQLite in-memory** and plain dataclasses — no PostgreSQL, no OpenAI, runs in under 2 seconds.

```
tests/unit/
├── test_db_models.py        18 tests  — ORM models + typed query functions
├── test_supervisor.py       15 tests  — all routing paths including skip logic
├── test_agents_week3.py     15 tests  — quality scoring formula, IF feature matrix
├── test_api_schemas.py      15 tests  — Pydantic v2 NPI validation, response shape
├── test_api_routes.py        9 tests  — FastAPI TestClient, mocked pipeline
└── test_agents_coverage.py  18 tests  — imputation, recommendations, fallbacks

tests/integration/
└── test_pipeline.py          9 tests  — full pipeline against live PostgreSQL
    ├── completes without error
    ├── quality score 0–100
    ├── anomaly flag (bool)
    ├── narrative > 20 chars
    ├── valid recommendation
    ├── agents tracked correctly
    ├── results written to database
    ├── completes under 30 seconds
    └── clean provider skips data cleaner
```

---

## Data

CareAgent uses **synthetic data** generated with the exact CMS Medicare Provider Utilization and Payment Data column structure (2022 release).

The synthetic generator replicates real CMS data characteristics:
- Log-normal payment and charge distributions
- Suppressed beneficiary counts (`< 11 patients → NaN`) matching CMS privacy rules
- 3% injected cost/quality anomalies for realistic Isolation Forest training
- 20 specialties · 20 states · 15 HCPCS procedure codes

**For production use**, download the real dataset from [data.cms.gov](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners) and run:

```bash
python scripts/load_cms_data.py --file data/cms_provider_2022.csv
```

---

## Limitations and Future Work

| Area | Current state | Planned |
|---|---|---|
| **LLM evaluation** | `narrative_faithfulness` and `narrative_relevancy` are `null` | RAGAS eval harness on held-out sample; log scores to MLflow |
| **Async execution** | DataCleaner and Anomaly run sequentially | `asyncio` parallel execution → ~2s pipeline |
| **Real CMS data** | Synthetic 10K providers | Full 9M-row dataset + OpenAI Batch API (~$12) |
| **LangSmith tracing** | Configured, requires API key | Full step-level trace on LangSmith free tier |
| **Scoring formula** | Proxy metrics (cost, volume, payment) | Clinical quality measures (readmission rates, HEDIS) |

---

## Resume Bullets

```
Architected LangGraph multi-agent system with dynamic supervisor routing across 5
specialized agents, processing CMS Medicare provider analysis in 0.36s avg with
92% test coverage across 99 tests — deployed on GCP Cloud Run

Built Isolation Forest anomaly detection identifying top 3% of outlier providers by
cost-quality ratio across 10,000 CMS Medicare provider records, confirmed 300/9,985
providers flagged at contamination=0.03

Implemented composite quality scoring formula benchmarked against specialty national
averages (cost efficiency 40% + volume percentile 30% + payment efficiency 30%) with
real-time PostgreSQL benchmark queries

Deployed multi-agent system on GCP Cloud Run with FastAPI serving scorecards in
0.36s, Neon PostgreSQL cloud database, and HuggingFace Spaces dashboard

Built GitHub Actions CI/CD pipeline with 90 unit tests + 85% coverage gate on every
push, using SQLite in-memory fixtures — zero external dependencies in CI
```

---

## Acknowledgements

Provider data structure modelled on [CMS Medicare Provider Utilization and Payment Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners) — free, public, used by real health systems.

---

## Author

**Ahmeduddin Mohammed**
- GitHub: [@MohammedAhmeduddin](https://github.com/MohammedAhmeduddin)
- LinkedIn: [linkedin.com/in/mohammed-ahmeduddin](https://www.linkedin.com/in/mohammed-ahmeduddin/)
- Portfolio: [SessionScout](https://huggingface.co/spaces/AhmeduddinMohammed/sessionscout) · [LoanLens](https://huggingface.co/spaces/AhmeduddinMohammed/loanlens) · [CareAgent](https://huggingface.co/spaces/AhmeduddinMohammed/careagent)

---

<div align="center">
<sub>Built with LangGraph · FastAPI · PostgreSQL · scikit-learn · GPT-4o-mini · GCP Cloud Run</sub>
</div>
