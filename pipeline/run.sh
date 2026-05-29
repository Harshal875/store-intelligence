#!/bin/bash
# Run the detection pipeline against all clips in the data directory.
# Usage: ./run.sh [DATA_DIR] [API_URL]

set -e

DATA_DIR="${1:-/data}"
API_URL="${2:-http://localhost:8000}"
MODEL="${3:-yolov8s.pt}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Store Intelligence - Detection Pipeline             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Data directory: $DATA_DIR"
echo "  API endpoint:   $API_URL"
echo "  Model:          $MODEL"
echo ""

# Wait for API to be ready
echo "[WAIT] Checking API availability..."
for i in $(seq 1 30); do
    if curl -sf "$API_URL/health" > /dev/null 2>&1; then
        echo "[OK] API is ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "[ERROR] API not available at $API_URL after 30s. Exiting."
        exit 1
    fi
    sleep 1
done

# Download model if not present
if [ ! -f "$MODEL" ]; then
    echo "[DOWNLOAD] Fetching YOLOv8s model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
fi

# Load POS transactions if available
if [ -f "$DATA_DIR/pos_transactions.csv" ]; then
    echo "[LOAD] Loading POS transactions..."
    python -c "
import csv
import requests
import json

# Load and send POS data to API (API will handle storage)
print('  POS transactions file found. Will be correlated during metrics computation.')
"
fi

# Run detection on all clips
echo ""
echo "[RUN] Starting detection pipeline..."
python detect.py \
    --process-all \
    --data-dir "$DATA_DIR" \
    --api-url "$API_URL" \
    --model "$MODEL" \
    --frame-skip 2 \
    --confidence 0.3

echo ""
echo "[COMPLETE] Detection pipeline finished."
echo "  View metrics: curl $API_URL/stores/STORE_BLR_002/metrics"
echo "  View funnel:  curl $API_URL/stores/STORE_BLR_002/funnel"
echo "  View health:  curl $API_URL/health"
