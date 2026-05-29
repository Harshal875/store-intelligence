"""Shared test fixtures for Store Intelligence API tests."""

import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.main import app
from app.database import Base, get_db, EventRecord, POSTransaction
from app.models import EventType


# Use SQLite for tests (in-memory)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Create tables before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Direct database session for test setup."""
    async with TestSessionLocal() as session:
        yield session


def make_event(
    store_id: str = "STORE_BLR_002",
    event_type: str = "ENTRY",
    visitor_id: str = None,
    zone_id: str = None,
    is_staff: bool = False,
    confidence: float = 0.9,
    dwell_ms: int = 0,
    timestamp: datetime = None,
    queue_depth: int = None,
) -> dict:
    """Helper to construct a valid event dict."""
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": zone_id,
            "session_seq": 1,
        },
    }


def make_event_batch(count: int, **kwargs) -> list[dict]:
    """Make a batch of events."""
    return [make_event(**kwargs) for _ in range(count)]


async def seed_pos_transaction(
    session: AsyncSession,
    store_id: str = "STORE_BLR_002",
    timestamp: datetime = None,
    amount: float = 1200.0,
):
    """Seed a POS transaction into the test database."""
    txn = POSTransaction(
        transaction_id=f"TXN_{uuid.uuid4().hex[:5]}",
        store_id=store_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        basket_value_inr=amount,
    )
    session.add(txn)
    await session.commit()
    return txn
