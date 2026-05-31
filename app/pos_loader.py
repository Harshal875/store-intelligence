"""POS transaction loader endpoint and startup ingestion."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone, date as _date

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import insert as sql_insert, text

from app.database import POSTransaction

logger = structlog.get_logger()


async def load_pos_from_csv(csv_path: str, db: AsyncSession) -> int:
    """Load POS transactions from the Brigade Bangalore CSV on startup."""
    if not os.path.exists(csv_path):
        return 0

    loaded = 0
    seen = set()
    # Aggregate total_amount per invoice (CSV has multiple line items per invoice)
    invoice_data: dict[str, dict] = {}

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                invoice = row.get("invoice_number", "").strip()
                if not invoice:
                    continue

                if invoice not in invoice_data:
                    date_str = row.get("order_date", "").strip()
                    time_str = row.get("order_time", "").strip()
                    try:
                        dt = datetime.strptime(f"{date_str} {time_str}", "%d-%m-%Y %H:%M:%S")
                        dt = dt.replace(tzinfo=timezone.utc)
                        # Shift to today so conversion queries (filtered by today) find them
                        today = datetime.now(timezone.utc).date()
                        dt = dt.replace(year=today.year, month=today.month, day=today.day)
                    except ValueError:
                        continue

                    store_id = row.get("store_id", "ST1008").strip() or "ST1008"
                    invoice_data[invoice] = {
                        "timestamp": dt,
                        "store_id": store_id,
                        "amount": 0.0,
                    }

                try:
                    amount = float(row.get("total_amount", 0) or 0)
                except ValueError:
                    amount = 0.0
                invoice_data[invoice]["amount"] += amount

        rows = [
            {
                "transaction_id": invoice,
                "store_id": data["store_id"],
                "timestamp": data["timestamp"],
                "basket_value_inr": round(data["amount"], 2),
            }
            for invoice, data in invoice_data.items()
        ]
        if rows:
            # Upsert: update timestamp/amount if invoice already exists (handles re-runs / date shifts)
            db_url = str(db.get_bind().url) if hasattr(db, "get_bind") else ""
            for row in rows:
                stmt = pg_insert(POSTransaction).values(**row).on_conflict_do_update(
                    index_elements=["transaction_id"],
                    set_={"timestamp": row["timestamp"], "basket_value_inr": row["basket_value_inr"]},
                )
                await db.execute(stmt)
                loaded += 1

        if loaded:
            await db.commit()
            logger.info("pos_transactions_loaded", count=loaded, path=csv_path)

    except Exception as e:
        logger.error("pos_load_failed", error=str(e))
        await db.rollback()

    return loaded
