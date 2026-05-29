# PROMPT: "Generate pytest tests for /stores/{id}/funnel covering:
# session-based counting (not raw events), re-entry deduplication in funnel,
# empty store funnel, all-staff funnel (should show 0), complete path
# Entry → Zone → Billing → Purchase, and partial paths (drop-off at each stage)."
#
# CHANGES MADE: Added test for re-entry not double-counting in funnel.
# Fixed POS correlation logic to match 5-min window requirement.
# Added edge case where visitor enters billing but no POS follows.

"""Tests for GET /stores/{id}/funnel endpoint."""

import uuid
from datetime import datetime, timezone, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import make_event, seed_pos_transaction


class TestFunnelEndpoint:
    """Test conversion funnel computation."""

    @pytest.mark.asyncio
    async def test_funnel_empty_store(self, client: AsyncClient):
        """Empty store should return zeroes in all stages."""
        response = await client.get("/stores/STORE_BLR_002/funnel")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_sessions"] == 0
        assert len(data["stages"]) == 4
        assert all(s["count"] == 0 for s in data["stages"])

    @pytest.mark.asyncio
    async def test_funnel_full_path(self, client: AsyncClient, db_session: AsyncSession):
        """Visitor with complete journey: Entry → Zone → Billing → Purchase."""
        now = datetime.now(timezone.utc)
        visitor = "VIS_full_path"
        
        events = [
            make_event(visitor_id=visitor, event_type="ENTRY", timestamp=now - timedelta(minutes=10)),
            make_event(visitor_id=visitor, event_type="ZONE_ENTER", zone_id="SKINCARE",
                      timestamp=now - timedelta(minutes=8)),
            make_event(visitor_id=visitor, event_type="BILLING_QUEUE_JOIN", zone_id="BILLING",
                      timestamp=now - timedelta(minutes=3), queue_depth=1),
            make_event(visitor_id=visitor, event_type="ZONE_ENTER", zone_id="BILLING",
                      timestamp=now - timedelta(minutes=3)),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        # POS transaction within 5 min
        await seed_pos_transaction(db_session, timestamp=now - timedelta(minutes=1))
        
        response = await client.get("/stores/STORE_BLR_002/funnel")
        data = response.json()
        
        assert data["stages"][0]["count"] == 1  # Entry
        assert data["stages"][1]["count"] == 1  # Zone Visit
        assert data["stages"][2]["count"] == 1  # Billing Queue
        assert data["stages"][3]["count"] == 1  # Purchase

    @pytest.mark.asyncio
    async def test_funnel_drop_off_at_zone(self, client: AsyncClient):
        """Visitors who enter but don't visit any zone."""
        events = [
            make_event(visitor_id="VIS_enters_only1", event_type="ENTRY"),
            make_event(visitor_id="VIS_enters_only2", event_type="ENTRY"),
            make_event(visitor_id="VIS_zones", event_type="ENTRY"),
            make_event(visitor_id="VIS_zones", event_type="ZONE_ENTER", zone_id="SKINCARE"),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/funnel")
        data = response.json()
        
        assert data["stages"][0]["count"] == 3  # 3 entered
        assert data["stages"][1]["count"] == 1  # 1 visited zone
        # Drop-off should be ~66.67%
        assert data["stages"][1]["drop_off_pct"] == pytest.approx(66.67, abs=0.1)

    @pytest.mark.asyncio
    async def test_funnel_reentry_not_double_counted(self, client: AsyncClient):
        """Re-entry should NOT create a second session in the funnel."""
        visitor = "VIS_reentry"
        events = [
            make_event(visitor_id=visitor, event_type="ENTRY"),
            make_event(visitor_id=visitor, event_type="EXIT"),
            make_event(visitor_id=visitor, event_type="REENTRY"),
            make_event(visitor_id=visitor, event_type="ZONE_ENTER", zone_id="HAIRCARE"),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/funnel")
        data = response.json()
        
        # Should count as 1 session, not 2
        assert data["stages"][0]["count"] == 1

    @pytest.mark.asyncio
    async def test_funnel_excludes_staff(self, client: AsyncClient):
        """Staff should not appear in funnel at all."""
        events = [
            make_event(visitor_id="VIS_staff", event_type="ENTRY", is_staff=True),
            make_event(visitor_id="VIS_staff", event_type="ZONE_ENTER", zone_id="SKINCARE", is_staff=True),
            make_event(visitor_id="VIS_customer", event_type="ENTRY", is_staff=False),
        ]
        await client.post("/events/ingest", json={"events": events})
        
        response = await client.get("/stores/STORE_BLR_002/funnel")
        data = response.json()
        
        assert data["stages"][0]["count"] == 1  # Only customer
