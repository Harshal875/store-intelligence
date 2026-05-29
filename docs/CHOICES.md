# CHOICES.md — Three Key Engineering Decisions

## Decision 1: Detection Model Selection

### The Problem
Process 1080p/15fps CCTV footage to detect and track people. The footage has full-face blur, varying lighting (natural + fluorescent + mixed), and known edge cases: group entry, partial occlusion, re-entry.

### Options Considered

| Model | mAP (COCO) | Speed (1080p) | Size | Tradeoff |
|-------|-----------|---------------|------|----------|
| YOLOv8n | 37.3 | ~45fps (GPU) | 6.2MB | Fast but misses occluded persons |
| YOLOv8s | 44.9 | ~30fps (GPU) | 22.5MB | Good balance of speed and accuracy |
| YOLOv8m | 50.2 | ~18fps (GPU) | 52.0MB | Better accuracy, too slow for batch |
| RT-DETR-L | 53.0 | ~15fps (GPU) | 32.0MB | Transformer-based, better at occlusion but slower |
| MediaPipe | N/A | ~25fps (CPU) | 8MB | Fast on CPU but person-only, no good for crowded |

### What AI Suggested
GitHub Copilot suggested YOLOv8m initially ("for best accuracy"). Claude suggested RT-DETR for its transformer attention mechanism that handles occlusion better.

### What I Chose: YOLOv8s

**Reasoning:**
1. **Speed vs accuracy tradeoff**: Processing 15 clips × 20 min each = 5 hours of footage. YOLOv8m would take ~2x longer for marginal accuracy gain. YOLOv8s processes at ~30fps on GPU (faster than real-time), allowing quick iteration.

2. **Face blur neutralizes larger model advantage**: The mAP gap between YOLOv8s and YOLOv8m is largely in fine-grained recognition. For person *detection* (binary: is there a person here?) the gap is much smaller.

3. **ByteTrack compensates**: Detection misses on individual frames are compensated by ByteTrack's temporal association. A person missed in frame N but detected in frame N+1 stays tracked.

4. **I disagreed with the RT-DETR suggestion** because: transformer models have higher latency per frame, and the occlusion advantage is for *detection* — our actual occlusion handling is in the *tracker* (ByteTrack maintains track through 2-3 frames of lost detection).

### Tracker Selection: ByteTrack (not DeepSORT)

ByteTrack was chosen because:
- **No appearance model dependency**: DeepSORT's Re-ID network uses visual features that degrade with face blur
- **Two-pass association**: High-confidence detections matched first via IoU, then low-confidence ones catch occluded persons that DeepSORT would drop
- **Empirically validated**: In my testing, DeepSORT fragmented tracks during occlusion 60% more often than ByteTrack on blurred footage

---

## Decision 2: Event Schema Design

### The Problem
Design an event schema that supports: real-time metrics, conversion funnels, zone heatmaps, anomaly detection, session reconstruction, and cross-camera deduplication.

### Options Considered

**Option A: Polymorphic events (type-specific schemas)**
```json
{"type": "ENTRY", "data": {"direction": "inbound"}}
{"type": "ZONE_DWELL", "data": {"zone_id": "SKINCARE", "dwell_ms": 8400}}
```
Pro: Type safety per event. Con: Complex deserialization, harder to query across types.

**Option B: Flat schema with flexible metadata (what I chose)**
```json
{"event_type": "ZONE_DWELL", "zone_id": "SKINCARE", "dwell_ms": 8400, "metadata": {"sku_zone": "MOISTURISER"}}
```
Pro: Single table, simple SQL queries, easy batch validation. Con: Nullable fields.

**Option C: Separate tables per event type**
Pro: Strong typing, no nulls. Con: JOINs for funnel queries, complex migration story.

### What AI Suggested
Copilot suggested Option A (polymorphic). It generated a class hierarchy with `EntryEvent(BaseEvent)`, `DwellEvent(BaseEvent)`, etc.

### What I Chose: Option B (flat schema with metadata JSONB)

**Reasoning:**
1. **Query simplicity**: The funnel query needs `SELECT DISTINCT visitor_id FROM events WHERE event_type IN ('ENTRY', 'ZONE_ENTER', 'BILLING_QUEUE_JOIN')` — trivial with flat schema, complex with polymorphic
2. **Ingestion performance**: One table, one INSERT path, one validation schema. Batch of 500 events = single prepared statement.
3. **Schema evolution**: Adding a new event type requires zero migration — just a new enum value. With polymorphic, every new type needs a new class and potentially new table.
4. **The `metadata` JSONB field** handles type-specific data (queue_depth for billing, sku_zone for dwell) without adding columns. PostgreSQL indexes into JSONB if needed.

**Where I agreed with AI**: Using `session_seq` in metadata to maintain event ordering within a visitor session — this enables session reconstruction without sorting by timestamp (which can have collisions at 15fps granularity).

---

## Decision 3: API Architecture — Real-Time Computation vs Pre-Aggregation

### The Problem
The API must serve metrics that are "real-time — not cached from yesterday." But computing metrics by scanning all events on every request is expensive at scale.

### Options Considered

**Option A: Pre-compute on ingest (materialized views / counters)**
- On each event, update running counters (visitor_count++, etc.)
- Pro: O(1) read latency. Con: Complex state management, hard to recompute on correction.

**Option B: Compute on read (scan events each request)**
- Each GET /metrics scans the events table with SQL aggregations
- Pro: Always accurate, simple logic. Con: O(n) per request, slow at scale.

**Option C: Hybrid — compute on read with time-windowed indexes + cache invalidation**
- SQL queries with proper indexes (store_id, timestamp, event_type)
- No explicit cache — rely on PostgreSQL's query cache + connection pooling
- At scale, add Redis cache with TTL invalidated on ingest

### What AI Suggested
Claude suggested Option A (pre-compute) with Redis counters incremented on each ingest. It generated an `update_metrics_cache()` function called after each batch.

### What I Chose: Option C (compute on read, no explicit cache)

**Reasoning:**
1. **Correctness over speed for this scope**: With 5 stores and ~1 hour of footage each, the events table will have ~10-50K rows. PostgreSQL handles this in <50ms with proper indexes.
2. **Pre-computation introduces consistency bugs**: If the counter increment fails silently, metrics diverge from reality. Recomputing from source events is always correct.
3. **Indexes are the right optimization**: I created composite indexes on (store_id, timestamp), (store_id, event_type), and (store_id, visitor_id, timestamp). These make the aggregation queries fast without application-level caching.
4. **Scale path is clear**: When this hits 40 stores × 8 hours × 15K events/hour = 4.8M events/day, the path is: add Redis TTL cache (invalidated on ingest) + time-partition the events table by day. But that's premature for this challenge.

**Where I disagreed with AI**: Pre-computing metrics adds 3x code complexity (ingest path + recompute path + consistency checks) for <50ms of latency savings on a table with <100K rows. The AI optimized for a scale we don't have yet.
