"""Health check endpoint logic."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, check_db_health
from app.models import HealthResponse, StoreHealth

logger = structlog.get_logger()

# Track application start time
APP_START_TIME = time.time()


async def get_health(db: AsyncSession) -> HealthResponse:
    """
    Return service health status.
    Includes last event timestamp per store and STALE_FEED warning if >10 min lag.
    """
    now = datetime.now(timezone.utc)
    uptime = time.time() - APP_START_TIME

    # Check database connectivity
    db_healthy = await check_db_health()
    if not db_healthy:
        return HealthResponse(
            status="unhealthy",
            timestamp=now,
            uptime_seconds=round(uptime, 2),
            stores=[],
        )

    # Get all stores and their last event timestamps
    store_status_q = select(
        EventRecord.store_id,
        func.max(EventRecord.timestamp).label("last_event"),
    ).group_by(EventRecord.store_id)

    result = await db.execute(store_status_q)
    store_data = result.fetchall()

    stores = []
    has_stale = False

    for row in store_data:
        store_id = row[0]
        last_event = row[1]
        
        if last_event is None:
            status = "NO_DATA"
            has_stale = True
        else:
            # Ensure timezone-aware (SQLite returns naive datetimes)
            if last_event.tzinfo is None:
                last_event = last_event.replace(tzinfo=timezone.utc)
            time_since = (now - last_event).total_seconds()
            if time_since > 600:  # 10 minutes
                status = "STALE_FEED"
                has_stale = True
            else:
                status = "HEALTHY"

        stores.append(StoreHealth(
            store_id=store_id,
            last_event_at=last_event,
            status=status,
        ))

    overall_status = "healthy"
    if has_stale:
        overall_status = "degraded"
    if not db_healthy:
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        timestamp=now,
        uptime_seconds=round(uptime, 2),
        stores=stores,
    )
