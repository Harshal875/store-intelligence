"""Zone detection using polygon intersection from store_layout.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Zone:
    """A named zone defined by a polygon in the camera's coordinate space."""
    zone_id: str
    polygon: np.ndarray  # Nx2 array of (x, y) vertices
    camera_id: str


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Ray casting algorithm to determine if a point is inside a polygon.
    polygon: Nx2 numpy array of vertices.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


class ZoneDetector:
    """Detect which zone a person is in based on their bounding box center."""

    def __init__(self, store_layout_path: str, store_id: str, camera_id: str):
        self.zones: list[Zone] = []
        self.store_id = store_id
        self.camera_id = camera_id
        self._load_zones(store_layout_path, store_id, camera_id)

    def _load_zones(self, path: str, store_id: str, camera_id: str):
        """Load zone polygons from store_layout.json."""
        try:
            with open(path, "r") as f:
                layout = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If layout not available, create default zones
            self._create_default_zones(camera_id)
            return

        # Try to find zones for this store and camera
        store_data = layout.get(store_id, layout.get("stores", {}).get(store_id, {}))
        if not store_data:
            self._create_default_zones(camera_id)
            return

        cameras = store_data.get("cameras", {})
        cam_data = cameras.get(camera_id, {})
        zones_data = cam_data.get("zones", store_data.get("zones", []))

        for zone_def in zones_data:
            if isinstance(zone_def, dict):
                zone_id = zone_def.get("zone_id", zone_def.get("name", "UNKNOWN"))
                polygon_data = zone_def.get("polygon", zone_def.get("coordinates", []))
                if polygon_data:
                    self.zones.append(Zone(
                        zone_id=zone_id,
                        polygon=np.array(polygon_data, dtype=np.float32),
                        camera_id=camera_id,
                    ))

    def _create_default_zones(self, camera_id: str):
        """Create sensible default zones based on camera type.
        Stores normalized coordinates (0-1 range). Call set_frame_size() to scale."""
        self._normalized = True
        if "ENTRY" in camera_id.upper():
            self.zones = [
                Zone("ENTRANCE", np.array([[0, 0.5], [1, 0.5], [1, 1], [0, 1]], dtype=np.float32), camera_id),
            ]
        elif "FLOOR" in camera_id.upper():
            self.zones = [
                Zone("SKINCARE", np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=np.float32), camera_id),
                Zone("HAIRCARE", np.array([[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]], dtype=np.float32), camera_id),
                Zone("FRAGRANCES", np.array([[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]], dtype=np.float32), camera_id),
                Zone("MAKEUP", np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]], dtype=np.float32), camera_id),
            ]
        elif "BILLING" in camera_id.upper():
            self.zones = [
                Zone("BILLING", np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), camera_id),
            ]

    def set_frame_size(self, width: int, height: int):
        """Scale normalized zone polygons to actual frame pixel dimensions."""
        if getattr(self, '_normalized', False):
            for zone in self.zones:
                scaled = zone.polygon.copy()
                scaled[:, 0] *= width
                scaled[:, 1] *= height
                zone.polygon = scaled
            self._normalized = False

    def detect_zone(self, bbox: np.ndarray) -> Optional[str]:
        """
        Given a bounding box [x1, y1, x2, y2], determine which zone
        the person's feet (bottom-center) are in.
        
        Uses bottom-center as it's more stable for perspective distortion.
        """
        # Bottom-center of bounding box (feet position)
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]  # Bottom of bbox
        
        for zone in self.zones:
            if point_in_polygon((foot_x, foot_y), zone.polygon):
                return zone.zone_id
        
        return None

    def get_all_zone_ids(self) -> list[str]:
        """Return all zone IDs for this camera."""
        return [z.zone_id for z in self.zones]
