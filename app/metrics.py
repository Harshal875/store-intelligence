"""Real-time metrics computation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select, func, distinct, case, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, POSTransaction
from app.models import StoreMetrics

logger = structlog.get_logger()


async def compute_store_metrics(
    store_id: str,
    db: AsyncSession,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> StoreMetrics:
    """
    Compute real-time metrics for a store.
    Excludes is_staff=true from all customer metrics.
    """
    now = datetime.now(timezone.utc)

    if window_start is None:
        # Default: today from midnight UTC
        window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if window_end is None:
        window_end = now

    # Base filter: store + time window + exclude staff
    base_filter = and_(
        EventRecord.store_id == store_id,
        EventRecord.timestamp >= window_start,
        EventRecord.timestamp <= window_end,
        EventRecord.is_staff == False,
    )

    # Unique visitors (distinct visitor_ids with ENTRY events)
    unique_visitors_q = select(
        func.count(distinct(EventRecord.visitor_id))
    ).where(
        and_(base_filter, EventRecord.event_type == "ENTRY")
    )
    result = await db.execute(unique_visitors_q)
    unique_visitors = result.scalar() or 0

    # Total entries and exits
    entries_q = select(func.count()).where(
        and_(base_filter, EventRecord.event_type == "ENTRY")
    )
    result = await db.execute(entries_q)
    total_entries = result.scalar() or 0

    exits_q = select(func.count()).where(
        and_(base_filter, EventRecord.event_type == "EXIT")
    )
    result = await db.execute(exits_q)
    total_exits = result.scalar() or 0

    # Current occupancy
    current_occupancy = max(0, total_entries - total_exits)

    # Conversion rate: visitors in billing zone within 5 min before a POS transaction
    # A visitor in billing zone in the 5-min window before a transaction = converted
    conversion_rate = 0.0
    if unique_visitors > 0:
        # Get POS transactions for this store in window
        pos_q = select(POSTransaction.timestamp).where(
            and_(
                POSTransaction.store_id == store_id,
                POSTransaction.timestamp >= window_start,
                POSTransaction.timestamp <= window_end,
            )
        )
        pos_result = await db.execute(pos_q)
        pos_timestamps = [row[0] for row in pos_result.fetchall()]

        if pos_timestamps:
            # Find distinct visitors in billing zone within 5 min before each transaction
            converted_visitors = set()
            for pos_ts in pos_timestamps:
                billing_window_start = pos_ts - timedelta(minutes=5)
                billing_q = select(distinct(EventRecord.visitor_id)).where(
                    and_(
                        EventRecord.store_id == store_id,
                        EventRecord.is_staff == False,
                        EventRecord.zone_id == "BILLING",
                        EventRecord.timestamp >= billing_window_start,
                        EventRecord.timestamp <= pos_ts,
                    )
                )
                billing_result = await db.execute(billing_q)
                for row in billing_result.fetchall():
                    converted_visitors.add(row[0])

            conversion_rate = len(converted_visitors) / unique_visitors if unique_visitors > 0 else 0.0

    # Average dwell per zone
    dwell_q = select(
        EventRecord.zone_id,
        func.avg(EventRecord.dwell_ms),
    ).where(
        and_(base_filter, EventRecord.event_type == "ZONE_DWELL", EventRecord.zone_id.isnot(None))
    ).group_by(EventRecord.zone_id)
    result = await db.execute(dwell_q)
    avg_dwell_per_zone = {row[0]: round(row[1], 2) for row in result.fetchall()}

    # Current queue depth (count visitors currently in billing zone)
    # Latest queue_depth from BILLING_QUEUE_JOIN events
    queue_q = select(EventRecord.metadata_json).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.timestamp >= window_start,
        )
    ).order_by(EventRecord.timestamp.desc()).limit(1)
    result = await db.execute(queue_q)
    queue_row = result.first()
    current_queue_depth = 0
    if queue_row and queue_row[0]:
        current_queue_depth = queue_row[0].get("queue_depth", 0) or 0

    # Abandonment rate
    abandon_q = select(func.count()).where(
        and_(base_filter, EventRecord.event_type == "BILLING_QUEUE_ABANDON")
    )
    result = await db.execute(abandon_q)
    abandonment_count = result.scalar() or 0

    join_q = select(func.count()).where(
        and_(base_filter, EventRecord.event_type == "BILLING_QUEUE_JOIN")
    )
    result = await db.execute(join_q)
    join_count = result.scalar() or 0

    abandonment_rate = abandonment_count / join_count if join_count > 0 else 0.0

    return StoreMetrics(
        store_id=store_id,
        timestamp=now,
        unique_visitors=unique_visitors,
        conversion_rate=round(conversion_rate, 4),
        avg_dwell_ms_per_zone=avg_dwell_per_zone,
        current_queue_depth=current_queue_depth,
        abandonment_rate=round(abandonment_rate, 4),
        total_entries=total_entries,
        total_exits=total_exits,
        current_occupancy=current_occupancy,
    )
