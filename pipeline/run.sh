#!/bin/bash
# Run the detection pipeline against all clips in the data directory.
# Usage: ./run.sh [DATA_DIR] [API_URL]

set -e

DATA_DIR="${1:-/data}"
API_URL="${2:-${API_URL:-http://localhost:8000}}"
MODEL="${3:-yolov8s.pt}"
STORE_ID="ST1008"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Store Intelligence - Detection Pipeline             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Data directory: $DATA_DIR"
echo "  API endpoint:   $API_URL"
echo "  Store ID:       $STORE_ID"
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
    echo "[DOWNLOAD] Fetching YOLOv8s model weights..."
    python -c "import torch; _o=torch.load; torch.load=lambda *a,**kw: _o(*a,**{**{'weights_only':False},**kw}); from ultralytics import YOLO; YOLO('yolov8s.pt')"
fi

# Load store layout and POS transactions
echo "[LOAD] Processing store layout and POS transactions..."
python load_data.py --data-dir "$DATA_DIR" --api-url "$API_URL" --store-id "$STORE_ID"

# Run detection on all clips
# CAM 1 = Entry/Exit (most important for entry counts)
# CAM 2-5 = Floor/Billing (zone tracking)
echo ""
echo "[RUN] Starting detection pipeline..."

# Process entry camera first (highest priority)
if [ -f "$DATA_DIR/clips/CAM 1.mp4" ]; then
    echo "  Processing entry camera (CAM 1)..."
    python detect.py \
        --video "$DATA_DIR/clips/CAM 1.mp4" \
        --store-id "$STORE_ID" \
        --camera-id "CAM_1_ENTRY" \
        --api-url "$API_URL" \
        --model "$MODEL"
fi

# Process remaining cameras
for cam in 2 3 4 5; do
    VIDEO="$DATA_DIR/clips/CAM ${cam}.mp4"
    if [ -f "$VIDEO" ]; then
        echo "  Processing CAM ${cam}..."
        CAM_TYPE="CAM_${cam}_FLOOR"
        if [ $cam -eq 4 ]; then
            CAM_TYPE="CAM_${cam}_BILLING"
        fi
        python detect.py \
            --video "$VIDEO" \
            --store-id "$STORE_ID" \
            --camera-id "$CAM_TYPE" \
            --api-url "$API_URL" \
            --model "$MODEL"
    fi
done

echo ""
echo "[COMPLETE] Detection pipeline finished."
echo "  View metrics:  curl $API_URL/stores/$STORE_ID/metrics | python -m json.tool"
echo "  View funnel:   curl $API_URL/stores/$STORE_ID/funnel  | python -m json.tool"
echo "  View anomalies: curl $API_URL/stores/$STORE_ID/anomalies | python -m json.tool"
echo "  Dashboard:     http://localhost:8000/dashboard"
