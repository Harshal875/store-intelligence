"""Store Intelligence API - FastAPI application entrypoint."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import init_db, get_db, check_db_health
from app.middleware import RequestLoggingMiddleware, setup_logging
from app.models import (
    IngestRequest, IngestResult, StoreMetrics,
    FunnelResponse, HeatmapResponse, AnomalyResponse, HealthResponse,
)
from app.ingestion import ingest_events
from app.metrics import compute_store_metrics
from app.funnel import compute_funnel
from app.heatmap import compute_heatmap
from app.anomalies import detect_anomalies
from app.health import get_health
from app.websocket import ws_manager

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    setup_logging()
    logger.info("starting_application")
    await init_db()
    logger.info("database_initialized")
    yield
    logger.info("shutting_down")


app = FastAPI(
    title="Store Intelligence API",
    description="Real-time store analytics from CCTV detection pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structured logging
app.add_middleware(RequestLoggingMiddleware)

# Serve dashboard static files
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """No raw stack traces in responses - graceful degradation."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "trace_id": getattr(request.state, "trace_id", None),
        },
    )


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Service status, last event timestamp per store, STALE_FEED warning."""
    try:
        return await get_health(db)
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "database_unavailable",
                "message": str(e),
            },
        )


# ─── Event Ingestion ──────────────────────────────────────────────────────────

@app.post("/events/ingest", response_model=IngestResult)
async def ingest(request: IngestRequest, db: AsyncSession = Depends(get_db)):
    """
    Ingest batch of up to 500 events.
    Idempotent by event_id. Partial success on malformed events.
    """
    db_ok = await check_db_health()
    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": "Database is unavailable. Please retry.",
            },
        )

    result = await ingest_events(request.events, db)

    # Broadcast to WebSocket clients
    for event in request.events:
        if not event.is_staff:
            await ws_manager.broadcast_event(
                event.store_id,
                event.event_type.value,
                {
                    "visitor_id": event.visitor_id,
                    "zone_id": event.zone_id,
                    "confidence": event.confidence,
                },
            )

    return result


# ─── Store Metrics ────────────────────────────────────────────────────────────

@app.get("/stores/{store_id}/metrics", response_model=StoreMetrics)
async def get_store_metrics(store_id: str, db: AsyncSession = Depends(get_db)):
    """
    Today's metrics: unique visitors, conversion rate, avg dwell per zone,
    queue depth, abandonment rate. Excludes staff. Real-time.
    """
    db_ok = await check_db_health()
    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "message": "Database unavailable"},
        )

    return await compute_store_metrics(store_id, db)


# ─── Funnel ───────────────────────────────────────────────────────────────────

@app.get("/stores/{store_id}/funnel", response_model=FunnelResponse)
async def get_store_funnel(store_id: str, db: AsyncSession = Depends(get_db)):
    """
    Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
    Session is the unit. Re-entries do not double-count.
    """
    db_ok = await check_db_health()
    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "message": "Database unavailable"},
        )

    return await compute_funnel(store_id, db)


# ─── Heatmap ──────────────────────────────────────────────────────────────────

@app.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse)
async def get_store_heatmap(store_id: str, db: AsyncSession = Depends(get_db)):
    """
    Zone visit frequency + avg dwell, normalised 0-100.
    Includes data_confidence flag if fewer than 20 sessions.
    """
    db_ok = await check_db_health()
    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "message": "Database unavailable"},
        )

    return await compute_heatmap(store_id, db)


# ─── Anomalies ────────────────────────────────────────────────────────────────

@app.get("/stores/{store_id}/anomalies", response_model=AnomalyResponse)
async def get_store_anomalies(store_id: str, db: AsyncSession = Depends(get_db)):
    """
    Active anomalies: queue spike, conversion drop vs 7-day avg, dead zone.
    """
    db_ok = await check_db_health()
    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "message": "Database unavailable"},
        )

    return await detect_anomalies(store_id, db)


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{store_id}")
async def websocket_endpoint(websocket: WebSocket, store_id: str):
    """Live metric updates for dashboard."""
    await ws_manager.connect(websocket, store_id)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, store_id)
