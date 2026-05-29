"""Database layer - SQLAlchemy models and connection management."""

from __future__ import annotations

import os
from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, Index,
    create_engine, event as sa_event
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://store_intel:store_intel_2026@localhost:5432/store_intelligence"
)

SYNC_DATABASE_URL = DATABASE_URL.replace("+asyncpg", "+psycopg2").replace("asyncpg", "psycopg2")

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    visitor_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    zone_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    dwell_ms: Mapped[int] = mapped_column(Integer, default=0)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_store_timestamp", "store_id", "timestamp"),
        Index("idx_store_event_type", "store_id", "event_type"),
        Index("idx_visitor_session", "store_id", "visitor_id", "timestamp"),
        Index("idx_zone_dwell", "store_id", "zone_id", "event_type"),
    )


class POSTransaction(Base):
    __tablename__ = "pos_transactions"

    transaction_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    basket_value_inr: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("idx_pos_store_time", "store_id", "timestamp"),
    )


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Dependency for FastAPI endpoints."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def check_db_health() -> bool:
    """Check if database is reachable."""
    try:
        async with async_session() as session:
            await session.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        return True
    except Exception:
        return False
