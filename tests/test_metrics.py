# PROMPT: "Generate pytest tests for /stores/{id}/metrics covering: empty store (no events),
# store with only staff events (should show 0 visitors), conversion rate calculation
# with POS data, zero-purchase store, average dwell per zone, queue depth tracking,
# and real-time computation (not cached)."
#
# CHANGES MADE: Added explicit POS transaction seeding for conversion rate test.
# Fixed timezone handling. Added test for metrics excluding staff.
# Added queue depth from metadata_json.

"""Tests for GET /stores/{id}/metrics endpoint."""

import uuid
from datetime import datetime, timezone, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import make_event, seed_pos_transaction


class TestMetricsEndpoint:
    """Test store metrics computation."""

    @pytest.mark.asyncio
    async def test_metrics_empty_store(self, client: AsyncClient):
        """Store with no events should return zero metrics, not crash."""
        response = await client.get("/stores/STORE_BLR_002/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["store_id"] == "STORE_BLR_002"
        assert data["unique_visitors"] == 0
        assert data["conversion_rate"] == 0.0
        assert data["current_occupancy"] == 0
        assert data["current_queue_depth"] == 0

    @pytest.mark.asyncio
    async def test_metrics_excludes_staff(self, client: AsyncClient):
        """Staff events should NOT count toward unique visitors."""
        # Ingest customer + staff events
        events = [
            make_event(visitor_id="VIS_cust01", is_staff=False),
            make_event(visitor_id="VIS_cust02", is_staff=False),
            make_event(visitor_id="VIS_staff01", is_staff=True),
            make_event(visitor_id="VIS_staff02", is_staff=True),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["unique_visitors"] == 2  # Only customers

    @pytest.mark.asyncio
    async def test_metrics_all_staff_clip(self, client: AsyncClient):
        """All-staff store should report 0 visitors."""
        events = [
            make_event(visitor_id=f"VIS_staff{i:02d}", is_staff=True)
            for i in range(5)
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/metrics")
        assert response.status_code == 200
        assert response.json()["unique_visitors"] == 0

    @pytest.mark.asyncio
    async def test_metrics_occupancy(self, client: AsyncClient):
        """Occupancy = entries - exits."""
        events = [
            make_event(visitor_id="VIS_a", event_type="ENTRY"),
            make_event(visitor_id="VIS_b", event_type="ENTRY"),
            make_event(visitor_id="VIS_c", event_type="ENTRY"),
            make_event(visitor_id="VIS_a", event_type="EXIT"),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/metrics")
        data = response.json()
        assert data["total_entries"] == 3
        assert data["total_exits"] == 1
        assert data["current_occupancy"] == 2

    @pytest.mark.asyncio
    async def test_metrics_conversion_rate(self, client: AsyncClient, db_session: AsyncSession):
        """Conversion rate: visitors in billing zone within 5 min before POS transaction."""
        now = datetime.now(timezone.utc)
        
        # Visitor in billing zone
        events = [
            make_event(visitor_id="VIS_buyer", event_type="ENTRY", timestamp=now - timedelta(minutes=3)),
            make_event(visitor_id="VIS_buyer", event_type="ZONE_ENTER", zone_id="BILLING", 
                      timestamp=now - timedelta(minutes=2)),
            make_event(visitor_id="VIS_browser", event_type="ENTRY", timestamp=now - timedelta(minutes=4)),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        # POS transaction at 'now'
        await seed_pos_transaction(db_session, timestamp=now)
        
        response = await client.get("/stores/STORE_BLR_002/metrics")
        data = response.json()
        # 1 converted out of 2 unique visitors = 0.5
        assert data["conversion_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_metrics_zero_purchases(self, client: AsyncClient):
        """Store with visitors but no POS transactions = 0% conversion."""
        events = [
            make_event(visitor_id="VIS_a", event_type="ENTRY"),
            make_event(visitor_id="VIS_b", event_type="ENTRY"),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/metrics")
        data = response.json()
        assert data["conversion_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_avg_dwell_per_zone(self, client: AsyncClient):
        """Average dwell should be computed per zone."""
        events = [
            make_event(visitor_id="VIS_a", event_type="ZONE_DWELL", zone_id="SKINCARE", dwell_ms=5000),
            make_event(visitor_id="VIS_b", event_type="ZONE_DWELL", zone_id="SKINCARE", dwell_ms=7000),
            make_event(visitor_id="VIS_c", event_type="ZONE_DWELL", zone_id="HAIRCARE", dwell_ms=3000),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/metrics")
        data = response.json()
        assert "SKINCARE" in data["avg_dwell_ms_per_zone"]
        assert data["avg_dwell_ms_per_zone"]["SKINCARE"] == 6000.0  # avg of 5000, 7000
        assert data["avg_dwell_ms_per_zone"]["HAIRCARE"] == 3000.0
