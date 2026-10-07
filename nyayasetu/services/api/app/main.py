"""Main FastAPI Application Entrypoint for NyayaSetu."""
from __future__ import annotations

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as api_v1_router
from app.core.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nyayasetu")

app = FastAPI(
    title="NyayaSetu API",
    description=(
        "AI Legal Guidance & Risk-Awareness Platform for Indian Civil Law. "
        "Provides decision support grounded in verified statutory sources."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware Configuration ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API Routers ───────────────────────────────────────────────────────
app.include_router(api_v1_router, prefix="/api")


@app.get("/", summary="Root Endpoint")
async def root() -> dict[str, str]:
    """Root welcome endpoint directing to API documentation."""
    return {
        "name": "NyayaSetu API",
        "version": "0.1.0",
        "description": "AI Legal Guidance & Risk-Awareness Platform for India",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
