"""Event emission - construct and send events to the API."""

from __future__ import annotations

import uuid
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from config import PipelineConfig


class EventEmitter:
    """Construct structured events and batch-send to the Intelligence API."""

    def __init__(self, config: PipelineConfig, store_id: str):
        self.config = config
        self.store_id = store_id
        self.buffer: list[dict] = []
        self.session_counters: dict[str, int] = {}  # visitor_id -> event count

    def _next_seq(self, visitor_id: str) -> int:
        """Get next session sequence number for a visitor."""
        self.session_counters[visitor_id] = self.session_counters.get(visitor_id, 0) + 1
        return self.session_counters[visitor_id]

    def _frame_to_timestamp(self, frame_idx: int, fps: float, video_start: datetime) -> str:
        """Convert frame index to ISO-8601 timestamp."""
        offset_seconds = frame_idx / fps
        ts = video_start + timedelta(seconds=offset_seconds)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def emit_entry(
        self,
        visitor_id: str,
        camera_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit ENTRY event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ENTRY",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": None,
                "sku_zone": None,
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def emit_exit(
        self,
        visitor_id: str,
        camera_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit EXIT event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "EXIT",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": None,
                "sku_zone": None,
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def emit_zone_enter(
        self,
        visitor_id: str,
        camera_id: str,
        zone_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit ZONE_ENTER event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ZONE_ENTER",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": zone_id,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": None,
                "sku_zone": zone_id,
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def emit_zone_exit(
        self,
        visitor_id: str,
        camera_id: str,
        zone_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        confidence: float,
        dwell_ms: int = 0,
        is_staff: bool = False,
    ):
        """Emit ZONE_EXIT event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ZONE_EXIT",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": None,
                "sku_zone": zone_id,
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def emit_zone_dwell(
        self,
        visitor_id: str,
        camera_id: str,
        zone_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        dwell_ms: int,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit ZONE_DWELL event (every 30s of continuous presence)."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "ZONE_DWELL",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": None,
                "sku_zone": zone_id,
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def emit_billing_queue_join(
        self,
        visitor_id: str,
        camera_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        queue_depth: int,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit BILLING_QUEUE_JOIN event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "BILLING_QUEUE_JOIN",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": "BILLING",
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone": "BILLING",
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def emit_billing_queue_abandon(
        self,
        visitor_id: str,
        camera_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit BILLING_QUEUE_ABANDON event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "BILLING_QUEUE_ABANDON",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": "BILLING",
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": None,
                "sku_zone": "BILLING",
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def emit_reentry(
        self,
        visitor_id: str,
        camera_id: str,
        frame_idx: int,
        fps: float,
        video_start: datetime,
        confidence: float,
        is_staff: bool = False,
    ):
        """Emit REENTRY event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": "REENTRY",
            "timestamp": self._frame_to_timestamp(frame_idx, fps, video_start),
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": is_staff,
            "confidence": round(confidence, 3),
            "metadata": {
                "queue_depth": None,
                "sku_zone": None,
                "session_seq": self._next_seq(visitor_id),
            },
        }
        self.buffer.append(event)
        self._maybe_flush()

    def _maybe_flush(self):
        """Flush buffer if it exceeds batch size."""
        if len(self.buffer) >= self.config.batch_size:
            self.flush()

    def flush(self):
        """Send buffered events to the API."""
        if not self.buffer:
            return

        url = f"{self.config.api_url}/events/ingest"
        payload = {"events": self.buffer}

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Flushed {len(self.buffer)} events → "
                      f"accepted={result['accepted']}, "
                      f"duplicates={result['duplicates']}, "
                      f"rejected={result['rejected']}")
            else:
                print(f"  ✗ Ingest failed: HTTP {response.status_code} - {response.text[:200]}")
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to API at {url}. Is it running?")
        except Exception as e:
            print(f"  ✗ Flush error: {e}")

        self.buffer = []

    def get_stats(self) -> dict:
        """Return emission statistics."""
        return {
            "buffered": len(self.buffer),
            "visitors_tracked": len(self.session_counters),
        }
