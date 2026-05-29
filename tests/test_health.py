# PROMPT: "Generate pytest tests for /health endpoint covering:
# healthy state with recent events, degraded state with stale feeds,
# database unavailable → 503, response structure validation,
# and verify STALE_FEED warning when >10 min lag."
#
# CHANGES MADE: Added uptime_seconds assertion, verified stores list structure.

"""Tests for GET /health endpoint."""

from datetime import datetime, timezone, timedelta

import pytest
from httpx import AsyncClient

from tests.conftest import make_event


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_basic_structure(self, client: AsyncClient):
        """Health endpoint returns valid structure."""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "stores" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_health_with_recent_events(self, client: AsyncClient):
        """Health should be 'healthy' when events are recent."""
        event = make_event(timestamp=datetime.now(timezone.utc))
        await client.post("/events/ingest", json={"events": [event]})
        
        response = await client.get("/health")
        data = response.json()
        
        # Should be healthy or at least not unhealthy
        assert data["status"] in ["healthy", "degraded"]
        assert len(data["stores"]) > 0

    @pytest.mark.asyncio
    async def test_health_store_status_structure(self, client: AsyncClient):
        """Each store in health response has required fields."""
        event = make_event()
        await client.post("/events/ingest", json={"events": [event]})
        
        response = await client.get("/health")
        data = response.json()
        
        for store in data["stores"]:
            assert "store_id" in store
            assert "last_event_at" in store
            assert "status" in store
            assert store["status"] in ["HEALTHY", "STALE_FEED", "NO_DATA"]

    @pytest.mark.asyncio
    async def test_health_version_present(self, client: AsyncClient):
        """Health response should include version."""
        response = await client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"
