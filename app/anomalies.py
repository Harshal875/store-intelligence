"""Anomaly detection - queue spike, conversion drop, dead zone, stale feed."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select, func, distinct, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, POSTransaction
from app.models import AnomalyResponse, Anomaly, AnomalySeverity, AnomalyType

logger = structlog.get_logger()


async def detect_anomalies(
    store_id: str,
    db: AsyncSession,
) -> AnomalyResponse:
    """
    Detect active anomalies for a store:
    - BILLING_QUEUE_SPIKE: queue depth significantly above average
    - CONVERSION_DROP: today's conversion rate < 7-day average by >30%
    - DEAD_ZONE: a zone with no visits in the last 30 minutes
    - STALE_FEED: no events received in >10 minutes
    """
    now = datetime.now(timezone.utc)
    anomalies = []

    # --- STALE_FEED ---
    last_event_q = select(func.max(EventRecord.timestamp)).where(
        EventRecord.store_id == store_id
    )
    result = await db.execute(last_event_q)
    last_event_time = result.scalar()

    if last_event_time:
        time_since_last = (now - last_event_time).total_seconds()
        if time_since_last > 600:  # 10 minutes
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                store_id=store_id,
                anomaly_type=AnomalyType.STALE_FEED,
                severity=AnomalySeverity.CRITICAL,
                detected_at=now,
                description=f"No events received for {int(time_since_last)}s (threshold: 600s)",
                suggested_action="Check camera feed connectivity and detection pipeline status",
                metadata={"seconds_since_last_event": int(time_since_last)},
            ))
    else:
        anomalies.append(Anomaly(
            anomaly_id=str(uuid.uuid4()),
            store_id=store_id,
            anomaly_type=AnomalyType.STALE_FEED,
            severity=AnomalySeverity.CRITICAL,
            detected_at=now,
            description="No events ever received for this store",
            suggested_action="Verify store configuration and camera pipeline deployment",
            metadata={},
        ))

    # --- BILLING_QUEUE_SPIKE ---
    # Compare current queue depth vs 1-hour rolling average
    one_hour_ago = now - timedelta(hours=1)
    queue_events_q = select(EventRecord.metadata_json).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
            EventRecord.timestamp >= one_hour_ago,
        )
    ).order_by(EventRecord.timestamp.desc())
    result = await db.execute(queue_events_q)
    queue_events = result.fetchall()

    if queue_events:
        queue_depths = [
            (row[0] or {}).get("queue_depth", 0) or 0
            for row in queue_events
        ]
        current_depth = queue_depths[0] if queue_depths else 0
        avg_depth = sum(queue_depths) / len(queue_depths) if queue_depths else 0

        if avg_depth > 0 and current_depth > avg_depth * 2 and current_depth >= 5:
            severity = AnomalySeverity.CRITICAL if current_depth > avg_depth * 3 else AnomalySeverity.WARN
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                store_id=store_id,
                anomaly_type=AnomalyType.BILLING_QUEUE_SPIKE,
                severity=severity,
                detected_at=now,
                description=f"Queue depth {current_depth} is {current_depth/avg_depth:.1f}x above hourly average ({avg_depth:.1f})",
                suggested_action="Open additional billing counter or deploy floor staff to manage queue",
                metadata={"current_depth": current_depth, "avg_depth": round(avg_depth, 1)},
            ))

    # --- CONVERSION_DROP ---
    # Compare today's conversion rate vs 7-day average
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seven_days_ago = today_start - timedelta(days=7)

    # Today's visitors
    today_visitors_q = select(func.count(distinct(EventRecord.visitor_id))).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= today_start,
        )
    )
    result = await db.execute(today_visitors_q)
    today_visitors = result.scalar() or 0

    # Today's transactions
    today_txn_q = select(func.count()).where(
        and_(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= today_start,
        )
    )
    result = await db.execute(today_txn_q)
    today_txns = result.scalar() or 0

    today_conversion = today_txns / today_visitors if today_visitors > 0 else 0.0

    # 7-day average (excluding today)
    week_visitors_q = select(func.count(distinct(EventRecord.visitor_id))).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= seven_days_ago,
            EventRecord.timestamp < today_start,
        )
    )
    result = await db.execute(week_visitors_q)
    week_visitors = result.scalar() or 0

    week_txn_q = select(func.count()).where(
        and_(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= seven_days_ago,
            POSTransaction.timestamp < today_start,
        )
    )
    result = await db.execute(week_txn_q)
    week_txns = result.scalar() or 0

    week_conversion = week_txns / week_visitors if week_visitors > 0 else 0.0

    if week_conversion > 0 and today_visitors >= 10:
        drop_pct = (week_conversion - today_conversion) / week_conversion
        if drop_pct > 0.3:  # >30% drop
            severity = AnomalySeverity.CRITICAL if drop_pct > 0.5 else AnomalySeverity.WARN
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                store_id=store_id,
                anomaly_type=AnomalyType.CONVERSION_DROP,
                severity=severity,
                detected_at=now,
                description=f"Conversion rate {today_conversion:.1%} is {drop_pct:.0%} below 7-day average ({week_conversion:.1%})",
                suggested_action="Review store layout changes, check if high-value product zones are adequately staffed",
                metadata={
                    "today_rate": round(today_conversion, 4),
                    "week_avg_rate": round(week_conversion, 4),
                    "drop_pct": round(drop_pct, 4),
                },
            ))

    # --- DEAD_ZONE ---
    # Zones with no visits in the last 30 minutes
    thirty_min_ago = now - timedelta(minutes=30)

    # Get all known zones for this store (from historical data)
    all_zones_q = select(distinct(EventRecord.zone_id)).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
        )
    )
    result = await db.execute(all_zones_q)
    all_zones = set(row[0] for row in result.fetchall())

    # Get zones with recent activity
    active_zones_q = select(distinct(EventRecord.zone_id)).where(
        and_(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
            EventRecord.timestamp >= thirty_min_ago,
            EventRecord.is_staff == False,
        )
    )
    result = await db.execute(active_zones_q)
    active_zones = set(row[0] for row in result.fetchall())

    dead_zones = all_zones - active_zones
    for zone in dead_zones:
        if zone == "BILLING":  # Billing zone being empty is not anomalous
            continue
        anomalies.append(Anomaly(
            anomaly_id=str(uuid.uuid4()),
            store_id=store_id,
            anomaly_type=AnomalyType.DEAD_ZONE,
            severity=AnomalySeverity.INFO,
            detected_at=now,
            description=f"Zone '{zone}' has had no visitor activity for 30+ minutes",
            suggested_action=f"Consider repositioning displays or signage to drive traffic to {zone}",
            metadata={"zone_id": zone},
        ))

    return AnomalyResponse(
        store_id=store_id,
        timestamp=now,
        anomalies=anomalies,
    )
