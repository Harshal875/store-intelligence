"""Zone heatmap computation."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import select, func, distinct, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord
from app.models import HeatmapResponse, HeatmapZone

logger = structlog.get_logger()


async def compute_heatmap(
    store_id: str,
    db: AsyncSession,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> HeatmapResponse:
    """
    Compute zone visit frequency + avg dwell, normalised 0-100.
    Include data_confidence flag if fewer than 20 sessions in window.
    """
    now = datetime.now(timezone.utc)

    if window_start is None:
        window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if window_end is None:
        window_end = now

    base_filter = and_(
        EventRecord.store_id == store_id,
        EventRecord.timestamp >= window_start,
        EventRecord.timestamp <= window_end,
        EventRecord.is_staff == False,
        EventRecord.zone_id.isnot(None),
    )

    # Get total unique sessions for confidence check
    session_q = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= window_start,
            EventRecord.timestamp <= window_end,
            EventRecord.is_staff == False,
            EventRecord.event_type == "ENTRY",
        )
    )
    result = await db.execute(session_q)
    total_sessions = result.scalar() or 0

    # Zone metrics: visit count + average dwell
    zone_q = select(
        EventRecord.zone_id,
        func.count(distinct(EventRecord.visitor_id)).label("visit_count"),
        func.avg(EventRecord.dwell_ms).label("avg_dwell"),
    ).where(
        and_(
            base_filter,
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]),
        )
    ).group_by(EventRecord.zone_id)

    result = await db.execute(zone_q)
    zone_data = result.fetchall()

    # Normalize scores 0-100
    zones = []
    max_visits = max((row[1] for row in zone_data), default=1)

    for row in zone_data:
        zone_id = row[0]
        visit_count = row[1]
        avg_dwell = row[2] or 0.0

        normalized_score = int((visit_count / max_visits) * 100) if max_visits > 0 else 0

        zone = HeatmapZone(
            zone_id=zone_id,
            visit_count=visit_count,
            avg_dwell_ms=round(avg_dwell, 2),
            normalized_score=normalized_score,
            data_confidence="LOW" if total_sessions < 20 else None,
        )
        zones.append(zone)

    return HeatmapResponse(
        store_id=store_id,
        timestamp=now,
        zones=zones,
    )
