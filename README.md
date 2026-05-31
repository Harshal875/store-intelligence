# Store Intelligence System

Real-time store analytics from CCTV footage. Processes raw camera feeds through a detection pipeline, emits structured events, and serves live metrics via a REST API.

## Quick Start (5 commands)

```bash
git clone <repo-url> && cd store-intelligence
cp .env.example .env

# Place dataset files in ./data/ before this step:
#   ./data/clips/CAM 1.mp4 ... CAM 5.mp4
#   ./data/pos_transactions.csv
#   ./data/store_layout.xlsx

docker compose up -d                    # Starts API + PostgreSQL + Redis
# POS transactions are auto-loaded on startup from ./data/pos_transactions.csv

docker compose --profile pipeline run pipeline  # Runs detection on all clips
```

The API is now live at `http://localhost:8000`. Dashboard at `http://localhost:8000/dashboard`.

Store ID for Brigade Road Bangalore: **ST1008**

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
curl http://localhost:8000/health | python -m json.tool

# Brigade Road store metrics (store_id = ST1008)
curl http://localhost:8000/stores/ST1008/metrics | python -m json.tool
curl http://localhost:8000/stores/ST1008/funnel  | python -m json.tool
curl http://localhost:8000/stores/ST1008/heatmap | python -m json.tool
curl http://localhost:8000/stores/ST1008/anomalies | python -m json.tool
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
