"""Event ingestion - validate, deduplicate, store."""

from __future__ import annotations

from datetime import datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import StoreEvent, IngestResult
from app.database import EventRecord

logger = structlog.get_logger()


async def ingest_events(
    events: list[StoreEvent],
    db: AsyncSession,
) -> IngestResult:
    """
    Ingest a batch of events. Idempotent by event_id.
    Returns counts of accepted, rejected (validation), and duplicates.
    """
    accepted = 0
    rejected = 0
    duplicates = 0
    errors = []

    # Check for duplicates in batch
    event_ids = [e.event_id for e in events]
    existing_query = select(EventRecord.event_id).where(
        EventRecord.event_id.in_(event_ids)
    )
    result = await db.execute(existing_query)
    existing_ids = set(row[0] for row in result.fetchall())

    # Also track duplicates within this batch
    seen_in_batch = set()

    for i, event in enumerate(events):
        # Duplicate check - within batch
        if event.event_id in seen_in_batch:
            duplicates += 1
            continue

        # Duplicate check - already in DB
        if event.event_id in existing_ids:
            duplicates += 1
            seen_in_batch.add(event.event_id)
            continue

        try:
            record = EventRecord(
                event_id=event.event_id,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                metadata_json=event.metadata.model_dump() if event.metadata else None,
                ingested_at=datetime.utcnow(),
            )
            db.add(record)
            accepted += 1
            seen_in_batch.add(event.event_id)
        except Exception as e:
            rejected += 1
            errors.append({
                "index": i,
                "event_id": event.event_id,
                "error": str(e),
            })

    if accepted > 0:
        try:
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error("batch_commit_failed", error=str(e), batch_size=len(events))
            # Retry individually
            accepted = 0
            for event in events:
                if event.event_id in existing_ids or event.event_id in seen_in_batch:
                    continue
                try:
                    record = EventRecord(
                        event_id=event.event_id,
                        store_id=event.store_id,
                        camera_id=event.camera_id,
                        visitor_id=event.visitor_id,
                        event_type=event.event_type.value,
                        timestamp=event.timestamp,
                        zone_id=event.zone_id,
                        dwell_ms=event.dwell_ms,
                        is_staff=event.is_staff,
                        confidence=event.confidence,
                        metadata_json=event.metadata.model_dump() if event.metadata else None,
                        ingested_at=datetime.utcnow(),
                    )
                    db.add(record)
                    await db.commit()
                    accepted += 1
                except Exception:
                    await db.rollback()
                    rejected += 1

    logger.info(
        "ingest_complete",
        accepted=accepted,
        rejected=rejected,
        duplicates=duplicates,
        batch_size=len(events),
    )

    return IngestResult(
        accepted=accepted,
        rejected=rejected,
        duplicates=duplicates,
        errors=errors,
    )
