"""
careagent.api.main
~~~~~~~~~~~~~~~~~~~
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from careagent.api.routes.analyze import router as analyze_router
from careagent.api.routes.health  import router as health_router

app = FastAPI(
    title="CareAgent — Provider Quality Intelligence API",
    description="Multi-agent system for automated provider quality scoring",
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(health_router)
