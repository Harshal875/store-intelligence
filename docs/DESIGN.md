# DESIGN.md — Store Intelligence System Architecture

## System Overview

The Store Intelligence System is an end-to-end pipeline that transforms raw CCTV footage into actionable retail analytics. The system is composed of four layers:

1. **Detection Layer** — Computer vision pipeline that processes video frames to detect, track, and classify people
2. **Event Layer** — Structured event stream emitted from detection to API
3. **Intelligence Layer** — REST API that ingests events, computes metrics, and detects anomalies
4. **Presentation Layer** — Live dashboard with WebSocket-driven real-time updates

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DETECTION PIPELINE                                   │
│                                                                               │
│  Video Frame → YOLOv8s (person detect) → ByteTrack (associate)               │
│       ↓                                                                       │
│  Virtual Line Crossing (entry/exit) ← Entry camera                           │
│  Zone Polygon Intersection ← Floor/Billing cameras                           │
│  Staff Classifier (heuristic + HSV color)                                    │
│  Re-ID Module (body appearance embeddings)                                   │
│       ↓                                                                       │
│  Event Emitter → HTTP POST /events/ingest (batched, max 500)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INTELLIGENCE API                                     │
│                                                                               │
│  FastAPI (async) → PostgreSQL (event store) + Redis (pub/sub)                │
│                                                                               │
│  Endpoints:                                                                   │
│    /events/ingest     — Idempotent batch ingestion                           │
│    /stores/{id}/metrics  — Real-time KPIs                                    │
│    /stores/{id}/funnel   — Session-based conversion funnel                   │
│    /stores/{id}/heatmap  — Zone heat scores (0-100)                          │
│    /stores/{id}/anomalies — Active operational anomalies                     │
│    /health            — Per-store feed freshness                              │
│    /ws/{store_id}     — WebSocket for live dashboard                         │
└─────────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LIVE DASHBOARD                                       │
│  Single-page HTML + vanilla JS                                               │
│  WebSocket connection for push-based metric updates                          │
│  Fallback to HTTP polling if WS unavailable                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why ByteTrack over DeepSORT?

The CCTV footage has full-face blur applied. DeepSORT's appearance model (typically a CNN trained on face/body features) relies on visual appearance for track association. With faces obscured, its re-identification accuracy degrades significantly.

ByteTrack uses IoU-based association in two passes (high-confidence first, then low-confidence). This makes it robust to appearance degradation because it tracks spatial continuity rather than visual similarity. It also has fewer hyperparameters and more deterministic behavior.

### Why PostgreSQL over SQLite?

SQLite's write lock means concurrent ingestion from multiple pipeline instances would serialize. With 5 stores × 3 cameras potentially feeding events simultaneously, PostgreSQL's row-level locking handles this correctly. The operational overhead is minimal (one line in docker-compose).

### Why Polygon-Based Zone Detection (not VLM)?

Zone classification could be done with a VLM (e.g., "which zone is this person in?") but introduces:
- Latency: 200-500ms per inference vs. <0.01ms for point-in-polygon
- Cost: API calls per detection
- Non-determinism: same input may give different zone labels

Since `store_layout.json` provides zone definitions, polygon intersection is deterministic, free, and instant. VLMs would only be useful if zone boundaries were ambiguous or unlabeled.

## Data Flow

1. Pipeline reads video frame (every 2nd frame for 15fps input → 7.5 effective fps)
2. YOLOv8s detects bounding boxes for class "person" with confidence ≥ 0.3
3. ByteTracker associates detections to existing tracks or creates new ones
4. For entry camera: check if track center crossed virtual line (direction determines ENTRY/EXIT)
5. For floor/billing cameras: point-in-polygon test on foot position (bbox bottom-center)
6. Staff classifier runs every ~30s: flags tracks with >10min presence + multi-zone coverage
7. Events are batched (max 100) and POSTed to the API
8. API validates, deduplicates by event_id, stores in PostgreSQL
9. WebSocket broadcasts to connected dashboard clients

## Concurrency Model

- API: async FastAPI with 2 uvicorn workers
- Database: async SQLAlchemy with connection pooling (20 connections, 10 overflow)
- Pipeline: sequential per-camera (parallelizable across cameras via multiple containers)
- Dashboard: WebSocket connections managed per-store with automatic cleanup

---

## AI-Assisted Decisions

### 1. Event Schema Design (Agreed with AI suggestion)

I asked GitHub Copilot to help design the event schema structure. It suggested a flat event model with an `event_type` enum and a flexible `metadata` JSONB field. I agreed because:
- Flat events are easier to query with SQL (no nested JOINs)
- The `metadata` field accommodates event-type-specific data (queue_depth for billing, sku_zone for dwell) without schema migration
- I would have over-engineered this with a polymorphic event model without the suggestion

### 2. Anomaly Detection Thresholds (Overrode AI suggestion)

Copilot suggested using statistical methods (Z-scores, IQR) for anomaly detection. I overrode this because:
- With only 1 hour of footage per store, we don't have enough data for meaningful statistical baselines
- Simple rule-based thresholds (>2x average for queue spike, >30% drop for conversion) are more interpretable and debuggable
- In production with 7+ days of data, I would switch to statistical methods, but for this challenge the rules are more defensible

### 3. Dashboard Architecture (Partially agreed)

AI suggested using React + Vite for the dashboard. I chose vanilla HTML/JS because:
- Zero build step = simpler Docker image
- The dashboard is a single page with 6 metrics — React is overkill
- WebSocket client in vanilla JS is 50 lines vs. a full React+hooks setup
- However, I took the AI's suggestion to use WebSocket (instead of SSE) because it enables bidirectional communication for future features (e.g., user subscribing to specific anomaly types)
