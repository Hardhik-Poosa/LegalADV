"""Unit tests for FastAPI endpoints using TestClient."""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.router import get_db, get_llm_client
from app.main import app
from app.services.answer_pipeline import LegalAnswerResponse

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "NyayaSetu API"
    assert "docs" in data


def test_health_check_endpoint():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_session.execute.return_value = mock_result

    mock_llm = MagicMock()
    mock_llm.health.return_value = True

    async def override_get_db():
        yield mock_session

    def override_get_llm():
        return mock_llm

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_client] = override_get_llm

    try:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["llm_service"] == "connected"
    finally:
        app.dependency_overrides.clear()


def test_ingest_endpoint_parse_error():
    payload = {
        "act_name": "Invalid Act",
        "raw_text": "This text has no numbered section headings anywhere.",
        "jurisdiction": "IN",
    }
    resp = client.post("/api/v1/ingest", json=payload)
    assert resp.status_code == 400
    assert "Failed to parse Act section structure" in resp.json()["detail"]
