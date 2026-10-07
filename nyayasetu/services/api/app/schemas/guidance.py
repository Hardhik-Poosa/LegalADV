"""Pydantic schemas for FastAPI guidance and ingestion endpoints."""
from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field


class GuidanceRequest(BaseModel):
    """Request payload for legal decision support and guidance."""

    query: str = Field(
        ...,
        min_length=3,
        description="Factual query or situation description from the user.",
    )
    jurisdiction: str = Field(
        default="IN",
        description="ISO 3166-1 alpha-2 or Indian state code (e.g. IN, IN-MH, IN-DL).",
    )
    as_of_date: datetime.date | None = Field(
        default=None,
        description="Optional effective date to check provision validity (defaults to today).",
    )


class IngestRequest(BaseModel):
    """Request payload for ingesting a new legal Act or statute."""

    act_name: str = Field(..., description="Canonical Act short name")
    raw_text: str = Field(..., min_length=20, description="Full raw statutory text")
    jurisdiction: str = Field(default="IN", description="Jurisdiction code")
    full_title: str | None = Field(default=None, description="Official long title")
    effective_from: datetime.date | None = Field(
        default=None, description="Effective date of Act"
    )
    version: int = Field(default=1, ge=1, description="Act version number")


class IngestResponse(BaseModel):
    """Response payload returned after successful Act ingestion."""

    document_id: int
    act_name: str
    jurisdiction: str
    version: int
    sections_loaded: int
    chunks_created: int
    checksum: str
