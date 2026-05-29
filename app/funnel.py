"""Conversion funnel computation - session-based."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import select, func, distinct, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, POSTransaction
from app.models import FunnelResponse, FunnelStage

logger = structlog.get_logger()


async def compute_funnel(
    store_id: str,
    db: AsyncSession,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> FunnelResponse:
    """
    Compute conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
    
    Unit is SESSION (visitor_id), not raw events.
    Re-entries must NOT double-count a visitor.
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
    )

    # Stage 1: Entry - distinct visitors who entered
    entry_q = select(distinct(EventRecord.visitor_id)).where(
        and_(base_filter, EventRecord.event_type.in_(["ENTRY", "REENTRY"]))
    )
    result = await db.execute(entry_q)
    entered_visitors = set(row[0] for row in result.fetchall())
    entry_count = len(entered_visitors)

    # Stage 2: Zone Visit - distinct visitors who entered any zone
    zone_q = select(distinct(EventRecord.visitor_id)).where(
        and_(
            base_filter,
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]),
            EventRecord.visitor_id.in_(entered_visitors) if entered_visitors else False,
        )
    )
    if entered_visitors:
        result = await db.execute(zone_q)
        zone_visitors = set(row[0] for row in result.fetchall())
    else:
        zone_visitors = set()
    zone_count = len(zone_visitors)

    # Stage 3: Billing Queue - distinct visitors who joined billing queue
    billing_q = select(distinct(EventRecord.visitor_id)).where(
        and_(
            base_filter,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.visitor_id.in_(entered_visitors) if entered_visitors else False,
        )
    )
    if entered_visitors:
        result = await db.execute(billing_q)
        billing_visitors = set(row[0] for row in result.fetchall())
    else:
        billing_visitors = set()
    billing_count = len(billing_visitors)

    # Stage 4: Purchase - visitors in billing zone within 5 min before a POS transaction
    purchase_count = 0
    if billing_visitors:
        pos_q = select(POSTransaction.timestamp).where(
            and_(
                POSTransaction.store_id == store_id,
                POSTransaction.timestamp >= window_start,
                POSTransaction.timestamp <= window_end,
            )
        )
        pos_result = await db.execute(pos_q)
        pos_timestamps = [row[0] for row in pos_result.fetchall()]

        purchased_visitors = set()
        for pos_ts in pos_timestamps:
            from datetime import timedelta
            billing_window_start = pos_ts - timedelta(minutes=5)
            conv_q = select(distinct(EventRecord.visitor_id)).where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.is_staff == False,
                    EventRecord.zone_id == "BILLING",
                    EventRecord.timestamp >= billing_window_start,
                    EventRecord.timestamp <= pos_ts,
                    EventRecord.visitor_id.in_(billing_visitors),
                )
            )
            conv_result = await db.execute(conv_q)
            for row in conv_result.fetchall():
                purchased_visitors.add(row[0])

        purchase_count = len(purchased_visitors)

    # Compute drop-off percentages
    stages = []

    stages.append(FunnelStage(
        stage="Entry",
        count=entry_count,
        drop_off_pct=0.0,
    ))

    stages.append(FunnelStage(
        stage="Zone Visit",
        count=zone_count,
        drop_off_pct=round(
            ((entry_count - zone_count) / entry_count * 100) if entry_count > 0 else 0.0, 2
        ),
    ))

    stages.append(FunnelStage(
        stage="Billing Queue",
        count=billing_count,
        drop_off_pct=round(
            ((zone_count - billing_count) / zone_count * 100) if zone_count > 0 else 0.0, 2
        ),
    ))

    stages.append(FunnelStage(
        stage="Purchase",
        count=purchase_count,
        drop_off_pct=round(
            ((billing_count - purchase_count) / billing_count * 100) if billing_count > 0 else 0.0, 2
        ),
    ))

    return FunnelResponse(
        store_id=store_id,
        timestamp=now,
        stages=stages,
        total_sessions=entry_count,
    )
