# PROMPT: "Generate pytest tests for /stores/{id}/heatmap covering:
# zone visit frequency normalisation 0-100, avg dwell computation,
# data_confidence flag when fewer than 20 sessions, empty store heatmap."
#
# CHANGES MADE: Added explicit normalisation assertion (max zone = 100).

"""Tests for GET /stores/{id}/heatmap endpoint."""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from tests.conftest import make_event


class TestHeatmapEndpoint:
    """Test zone heatmap computation."""

    @pytest.mark.asyncio
    async def test_heatmap_empty_store(self, client: AsyncClient):
        """Empty store returns empty zones list."""
        response = await client.get("/stores/STORE_BLR_002/heatmap")
        
        assert response.status_code == 200
        data = response.json()
        assert data["store_id"] == "STORE_BLR_002"
        assert data["zones"] == []

    @pytest.mark.asyncio
    async def test_heatmap_normalisation(self, client: AsyncClient):
        """Most visited zone should have normalized_score = 100."""
        events = [
            # SKINCARE: 3 visitors
            make_event(visitor_id="VIS_a", event_type="ZONE_ENTER", zone_id="SKINCARE"),
            make_event(visitor_id="VIS_b", event_type="ZONE_ENTER", zone_id="SKINCARE"),
            make_event(visitor_id="VIS_c", event_type="ZONE_ENTER", zone_id="SKINCARE"),
            # HAIRCARE: 1 visitor
            make_event(visitor_id="VIS_d", event_type="ZONE_ENTER", zone_id="HAIRCARE"),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/heatmap")
        data = response.json()
        
        zones_by_id = {z["zone_id"]: z for z in data["zones"]}
        assert zones_by_id["SKINCARE"]["normalized_score"] == 100
        assert zones_by_id["HAIRCARE"]["normalized_score"] < 100

    @pytest.mark.asyncio
    async def test_heatmap_low_confidence_flag(self, client: AsyncClient):
        """Fewer than 20 sessions should set data_confidence = 'LOW'."""
        # Only 5 unique visitors entering
        events = [
            make_event(visitor_id=f"VIS_{i}", event_type="ENTRY") for i in range(5)
        ] + [
            make_event(visitor_id="VIS_0", event_type="ZONE_ENTER", zone_id="SKINCARE"),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/heatmap")
        data = response.json()
        
        for zone in data["zones"]:
            assert zone["data_confidence"] == "LOW"

    @pytest.mark.asyncio
    async def test_heatmap_excludes_staff(self, client: AsyncClient):
        """Staff zone visits should not appear in heatmap."""
        events = [
            make_event(visitor_id="VIS_staff", event_type="ZONE_ENTER", zone_id="SKINCARE", is_staff=True),
            make_event(visitor_id="VIS_cust", event_type="ZONE_ENTER", zone_id="HAIRCARE", is_staff=False),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/heatmap")
        data = response.json()
        
        zone_ids = [z["zone_id"] for z in data["zones"]]
        # SKINCARE should not appear (only staff visited it)
        assert "SKINCARE" not in zone_ids
        assert "HAIRCARE" in zone_ids
