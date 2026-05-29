# Store Intelligence System

Real-time store analytics from CCTV footage. Processes raw camera feeds through a detection pipeline, emits structured events, and serves live metrics via a REST API.

## Quick Start (5 commands)

```bash
git clone <repo-url> && cd store-intelligence
cp .env.example .env
docker compose up -d                    # Starts API + PostgreSQL + Redis
# Place video clips in ./data/ directory
docker compose --profile pipeline run pipeline  # Runs detection on clips
```

The API is now live at `http://localhost:8000`. Dashboard at `http://localhost:8000/dashboard`.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  CCTV Clips  │ ──→ │  Detection   │ ──→ │  Intelligence│ ──→ │  Dashboard   │
│  (raw video) │     │  Pipeline    │     │  API         │     │  (WebSocket) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                      YOLOv8s + ByteTrack   FastAPI + PostgreSQL   Live metrics
```

## Components

### Detection Pipeline (`pipeline/`)
- **YOLOv8s** for person detection
- **ByteTrack** for multi-object tracking (appearance-model-free, handles face blur)
- **Virtual line crossing** for entry/exit counting
- **Polygon intersection** for zone detection
- **Staff detection** via temporal heuristic + HSV color clustering
- **Re-ID** via body appearance embeddings for re-entry detection

### Intelligence API (`app/`)
- `POST /events/ingest` — Batch event ingestion (idempotent, up to 500/batch)
- `GET /stores/{id}/metrics` — Real-time visitors, conversion, dwell, queue
- `GET /stores/{id}/funnel` — Conversion funnel (session-based, deduped)
- `GET /stores/{id}/heatmap` — Zone visit frequency, normalised 0-100
- `GET /stores/{id}/anomalies` — Queue spike, conversion drop, dead zone, stale feed
- `GET /health` — Service status with per-store feed freshness

### Live Dashboard (`dashboard/`)
- WebSocket-connected real-time metrics
- Conversion funnel visualisation
- Live event feed
- Accessible at `http://localhost:8000/dashboard`

## Running the Detection Pipeline

### Against all clips:
```bash
docker compose --profile pipeline run pipeline --process-all --data-dir /data
```

### Against a specific clip:
```bash
docker compose --profile pipeline run pipeline \
  --video /data/clips/store_blr_002_entry.mp4 \
  --store-id STORE_BLR_002 \
  --camera-id CAM_ENTRY_01
```

### Without Docker (local development):
```bash
cd pipeline
pip install -r requirements.txt
python detect.py --process-all --data-dir ../data --api-url http://localhost:8000
```

## Running Tests

```bash
pip install -r requirements.txt -r requirements-test.txt
pytest --cov=app --cov-report=term-missing
```

## API Examples

```bash
# Check health
curl http://localhost:8000/health | jq

# Get store metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics | jq

# Get conversion funnel
curl http://localhost:8000/stores/STORE_BLR_002/funnel | jq

# Get anomalies
curl http://localhost:8000/stores/STORE_BLR_002/anomalies | jq

# Ingest events manually
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [{"event_id": "550e8400-e29b-41d4-a716-446655440000", "store_id": "STORE_BLR_002", "camera_id": "CAM_ENTRY_01", "visitor_id": "VIS_abc123", "event_type": "ENTRY", "timestamp": "2026-03-03T14:22:10Z", "zone_id": null, "dwell_ms": 0, "is_staff": false, "confidence": 0.92, "metadata": {"queue_depth": null, "sku_zone": null, "session_seq": 1}}]}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgresql+asyncpg://... | PostgreSQL connection string |
| `REDIS_HOST` | redis | Redis hostname |
| `CONFIDENCE_THRESHOLD` | 0.3 | Min detection confidence |
| `BATCH_SIZE` | 100 | Events per API batch |
| `FRAME_SKIP` | 2 | Process every Nth frame |

## Edge Cases Handled

- **Group entry**: Individual bounding boxes counted separately
- **Staff movement**: Excluded from all customer metrics via heuristic + color classification
- **Re-entry**: Same visitor re-entering produces REENTRY event, not duplicate ENTRY
- **Partial occlusion**: Low-confidence events emitted (not suppressed)
- **Billing queue**: Queue depth tracking with abandonment detection
- **Empty store**: Zero-state handled gracefully (no crashes, no nulls)
- **Camera overlap**: Cross-camera deduplication via shared visitor state
