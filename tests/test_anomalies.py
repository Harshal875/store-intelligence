# PROMPT: "Generate pytest tests for /stores/{id}/anomalies covering:
# STALE_FEED detection (no events in >10 min), DEAD_ZONE detection (zone with
# no activity in 30 min), BILLING_QUEUE_SPIKE, CONVERSION_DROP vs 7-day average,
# and verify anomaly severity levels and suggested_action fields."
#
# CHANGES MADE: Added test for store with no events at all (should return STALE_FEED).
# Added test verifying anomaly structure matches schema.
# Fixed time mocking to properly simulate stale feeds.

"""Tests for GET /stores/{id}/anomalies endpoint."""

import uuid
from datetime import datetime, timezone, timedelta

import pytest
from httpx import AsyncClient

from tests.conftest import make_event


class TestAnomaliesEndpoint:
    """Test anomaly detection logic."""

    @pytest.mark.asyncio
    async def test_anomalies_no_events(self, client: AsyncClient):
        """Store with no events should trigger STALE_FEED anomaly."""
        response = await client.get("/stores/STORE_BLR_002/anomalies")
        
        assert response.status_code == 200
        data = response.json()
        assert data["store_id"] == "STORE_BLR_002"
        
        # Should have at least one anomaly: STALE_FEED
        anomaly_types = [a["anomaly_type"] for a in data["anomalies"]]
        assert "STALE_FEED" in anomaly_types

    @pytest.mark.asyncio
    async def test_anomaly_structure(self, client: AsyncClient):
        """Verify anomaly response structure matches required schema."""
        response = await client.get("/stores/STORE_BLR_002/anomalies")
        data = response.json()
        
        for anomaly in data["anomalies"]:
            assert "anomaly_id" in anomaly
            assert "store_id" in anomaly
            assert "anomaly_type" in anomaly
            assert "severity" in anomaly
            assert "detected_at" in anomaly
            assert "description" in anomaly
            assert "suggested_action" in anomaly
            assert anomaly["severity"] in ["INFO", "WARN", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_stale_feed_severity_is_critical(self, client: AsyncClient):
        """STALE_FEED should be CRITICAL severity."""
        response = await client.get("/stores/STORE_BLR_002/anomalies")
        data = response.json()
        
        stale_anomalies = [a for a in data["anomalies"] if a["anomaly_type"] == "STALE_FEED"]
        assert len(stale_anomalies) > 0
        assert stale_anomalies[0]["severity"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_dead_zone_detection(self, client: AsyncClient):
        """Zone with no activity in 30+ min should trigger DEAD_ZONE."""
        # Seed events from >30 min ago in two zones
        old_time = datetime.now(timezone.utc) - timedelta(minutes=45)
        
        events = [
            make_event(visitor_id="VIS_a", event_type="ZONE_ENTER", zone_id="SKINCARE",
                      timestamp=old_time),
            make_event(visitor_id="VIS_b", event_type="ZONE_ENTER", zone_id="HAIRCARE",
                      timestamp=old_time),
            # Only SKINCARE has recent activity
            make_event(visitor_id="VIS_c", event_type="ZONE_ENTER", zone_id="SKINCARE",
                      timestamp=datetime.now(timezone.utc) - timedelta(minutes=5)),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/anomalies")
        data = response.json()
        
        dead_zones = [a for a in data["anomalies"] if a["anomaly_type"] == "DEAD_ZONE"]
        dead_zone_ids = [a["metadata"].get("zone_id") for a in dead_zones]
        assert "HAIRCARE" in dead_zone_ids

    @pytest.mark.asyncio
    async def test_no_false_positive_when_recent(self, client: AsyncClient):
        """Recent events should NOT trigger STALE_FEED."""
        # Ingest a recent event
        event = make_event(timestamp=datetime.now(timezone.utc))
        await client.post("/events/ingest", json={"events": [event]})
        
        response = await client.get("/stores/STORE_BLR_002/anomalies")
        data = response.json()
        
        stale_anomalies = [a for a in data["anomalies"] if a["anomaly_type"] == "STALE_FEED"]
        assert len(stale_anomalies) == 0

    @pytest.mark.asyncio
    async def test_suggested_action_not_empty(self, client: AsyncClient):
        """All anomalies must have non-empty suggested_action."""
        response = await client.get("/stores/STORE_BLR_002/anomalies")
        data = response.json()
        
        for anomaly in data["anomalies"]:
            assert anomaly["suggested_action"]
            assert len(anomaly["suggested_action"]) > 10
