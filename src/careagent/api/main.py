"""
careagent.api.main
~~~~~~~~~~~~~~~~~~~
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from careagent.api.routes.analyze import router as analyze_router
from careagent.api.routes.health import router as health_router
from careagent.db.queries import get_example_npis
from careagent.db.session import get_db
from careagent.scoring.peer import DISCLAIMER

TITLE = "CareAgent: Medicare Provider Billing-Pattern Analysis API"
DESCRIPTION = (
    "Multi-agent (LangGraph) comparison of Medicare billing patterns against "
    "same-specialty peers, using CMS 2024 data for New Jersey. " + DISCLAIMER
)

app = FastAPI(title=TITLE, description=DESCRIPTION, version="2.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    try:
        with get_db() as db:
            examples = get_example_npis(db)
    except Exception:
        logger.exception("Could not load example NPIs")
        examples = {}
    return JSONResponse({
        "name": TITLE,
        "version": "2.0.0",
        "data": "CMS Medicare Physician & Other Practitioners - by Provider, 2024, New Jersey",
        "disclaimer": DISCLAIMER,
        "endpoints": {"docs": "/docs", "health": "/health", "analyze": "POST /analyze"},
        "example_npis": examples,
        "github": "https://github.com/MohammedAhmeduddin/careagent",
    })


app.include_router(analyze_router)
app.include_router(health_router)
