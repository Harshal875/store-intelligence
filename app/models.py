"""Store Intelligence API - Pydantic models and schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None


class StoreEvent(BaseModel):
    event_id: str = Field(..., description="UUID v4 - globally unique")
    store_id: str = Field(..., description="Store identifier from store_layout.json")
    camera_id: str = Field(..., description="Camera that produced this event")
    visitor_id: str = Field(..., description="Re-ID token - unique per visit session")
    event_type: EventType
    timestamp: datetime = Field(..., description="ISO-8601 UTC")
    zone_id: Optional[str] = Field(None, description="Zone name; null for ENTRY/EXIT")
    dwell_ms: int = Field(0, ge=0, description="Duration in ms; 0 for instantaneous")
    is_staff: bool = Field(False, description="Whether this person is staff")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        try:
            uuid.UUID(v, version=4)
        except ValueError:
            raise ValueError("event_id must be a valid UUID v4")
        return v

    @field_validator("store_id")
    @classmethod
    def validate_store_id(cls, v: str) -> str:
        # Accept both STORE_XXX and ST#### formats
        if not (v.startswith("STORE_") or v.startswith("ST")):
            raise ValueError("store_id must start with 'STORE_' or 'ST'")
        return v


class IngestRequest(BaseModel):
    events: list[StoreEvent] = Field(..., max_length=500)


class IngestResult(BaseModel):
    accepted: int
    rejected: int
    duplicates: int
    errors: list[dict] = Field(default_factory=list)


class StoreMetrics(BaseModel):
    store_id: str
    timestamp: datetime
    unique_visitors: int
    conversion_rate: float
    avg_dwell_ms_per_zone: dict[str, float]
    current_queue_depth: int
    abandonment_rate: float
    total_entries: int
    total_exits: int
    current_occupancy: int


class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    timestamp: datetime
    stages: list[FunnelStage]
    total_sessions: int


class HeatmapZone(BaseModel):
    zone_id: str
    visit_count: int
    avg_dwell_ms: float
    normalized_score: int = Field(..., ge=0, le=100)
    data_confidence: Optional[str] = None


class HeatmapResponse(BaseModel):
    store_id: str
    timestamp: datetime
    zones: list[HeatmapZone]


class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class AnomalyType(str, Enum):
    BILLING_QUEUE_SPIKE = "BILLING_QUEUE_SPIKE"
    CONVERSION_DROP = "CONVERSION_DROP"
    DEAD_ZONE = "DEAD_ZONE"
    STALE_FEED = "STALE_FEED"


class Anomaly(BaseModel):
    anomaly_id: str
    store_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    detected_at: datetime
    description: str
    suggested_action: str
    metadata: dict = Field(default_factory=dict)


class AnomalyResponse(BaseModel):
    store_id: str
    timestamp: datetime
    anomalies: list[Anomaly]


class StoreHealth(BaseModel):
    store_id: str
    last_event_at: Optional[datetime]
    status: str  # "HEALTHY", "STALE_FEED", "NO_DATA"


class HealthResponse(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    uptime_seconds: float
    stores: list[StoreHealth]
    version: str = "1.0.0"
