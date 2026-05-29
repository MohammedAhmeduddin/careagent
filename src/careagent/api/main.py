"""
careagent.api.main
~~~~~~~~~~~~~~~~~~~
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "name":        "CareAgent — Provider Quality Intelligence API",
        "version":     "0.1.0",
        "description": "Multi-agent system for automated Medicare provider quality scoring",
        "endpoints": {
            "docs":     "/docs",
            "health":   "/health",
            "analyze":  "POST /analyze"
        },
        "demo_npis": {
            "include_example": "1000153386",
            "anomaly_example": "5133794489",
            "review_example":  "6237376063"
        },
        "github": "https://github.com/MohammedAhmeduddin/careagent"
    })


app.include_router(analyze_router)
app.include_router(health_router)
