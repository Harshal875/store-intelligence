"""Pipeline configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """Detection pipeline configuration."""
    
    # API endpoint
    api_url: str = os.getenv("API_URL", "http://localhost:8000")
    
    # Detection
    model_path: str = os.getenv("MODEL_PATH", "yolov8s.pt")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
    person_class_id: int = 0  # COCO class ID for 'person'
    
    # Tracking
    track_buffer: int = 30  # frames to keep lost tracks
    match_threshold: float = 0.8  # IoU threshold for matching
    
    # Zone detection
    store_layout_path: str = os.getenv("STORE_LAYOUT", "/data/store_layout.json")
    
    # Staff detection
    staff_dwell_threshold_ms: int = int(os.getenv("STAFF_DWELL_THRESHOLD_MS", "600000"))  # 10 min
    staff_zone_count_threshold: int = 3  # seen in 3+ zones → likely staff
    
    # Re-entry detection
    reentry_window_ms: int = int(os.getenv("REENTRY_WINDOW_MS", "300000"))  # 5 min
    reentry_similarity_threshold: float = 0.7
    
    # Event emission
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
    dwell_emit_interval_ms: int = 30000  # emit ZONE_DWELL every 30s
    
    # Video processing
    frame_skip: int = int(os.getenv("FRAME_SKIP", "2"))  # process every Nth frame
    
    # Entry/exit line configuration (normalized coordinates)
    # Will be overridden from store_layout.json per camera
    entry_line_y: float = 0.7  # Percentage of frame height for virtual line
    entry_direction: str = "down"  # "down" = entry, "up" = exit


@dataclass
class StoreConfig:
    """Per-store configuration loaded from store_layout.json."""
    store_id: str
    cameras: dict = field(default_factory=dict)
    zones: list = field(default_factory=list)
    open_hours: dict = field(default_factory=dict)
