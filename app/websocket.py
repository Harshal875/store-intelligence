"""WebSocket endpoint for live dashboard updates."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Set

import structlog
from fastapi import WebSocket, WebSocketDisconnect

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections for live metric updates."""

    def __init__(self):
        self.active_connections: dict[str, Set[WebSocket]] = {}  # store_id -> connections

    async def connect(self, websocket: WebSocket, store_id: str):
        await websocket.accept()
        if store_id not in self.active_connections:
            self.active_connections[store_id] = set()
        self.active_connections[store_id].add(websocket)
        logger.info("ws_connected", store_id=store_id, total=len(self.active_connections[store_id]))

    def disconnect(self, websocket: WebSocket, store_id: str):
        if store_id in self.active_connections:
            self.active_connections[store_id].discard(websocket)
            logger.info("ws_disconnected", store_id=store_id)

    async def broadcast_to_store(self, store_id: str, data: dict):
        """Broadcast a metric update to all connections watching a store."""
        if store_id not in self.active_connections:
            return
        
        message = json.dumps(data, default=str)
        disconnected = set()
        
        for connection in self.active_connections[store_id]:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.add(connection)
        
        # Clean up dead connections
        for conn in disconnected:
            self.active_connections[store_id].discard(conn)

    async def broadcast_event(self, store_id: str, event_type: str, payload: dict):
        """Broadcast a new event notification."""
        await self.broadcast_to_store(store_id, {
            "type": "event",
            "event_type": event_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def broadcast_metrics(self, store_id: str, metrics: dict):
        """Broadcast updated metrics."""
        await self.broadcast_to_store(store_id, {
            "type": "metrics_update",
            "payload": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# Global connection manager
ws_manager = ConnectionManager()
