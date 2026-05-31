"""POS transaction loader endpoint and startup ingestion."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

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

        for invoice, data in invoice_data.items():
            txn = POSTransaction(
                transaction_id=invoice,
                store_id=data["store_id"],
                timestamp=data["timestamp"],
                basket_value_inr=round(data["amount"], 2),
            )
            db.add(txn)
            loaded += 1

        if loaded:
            await db.commit()
            logger.info("pos_transactions_loaded", count=loaded, path=csv_path)

    except Exception as e:
        logger.error("pos_load_failed", error=str(e))
        await db.rollback()

    return loaded
