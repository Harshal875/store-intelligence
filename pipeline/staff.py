"""Staff detection heuristics + color-based classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2


@dataclass
class StaffClassifier:
    """
    Classifies whether a tracked person is staff using two signals:
    
    1. Temporal heuristic: Person present continuously from near store-open,
       traverses multiple zones, never exits → likely staff.
       
    2. HSV color clustering: Staff uniforms have consistent color signatures.
       Cluster dominant torso colors; the cluster that appears at store-open
       and persists throughout = staff uniform.
    """
    
    # Heuristic thresholds
    min_presence_frames: int = 9000  # ~10 min at 15fps
    min_zones_visited: int = 3
    
    # Color clustering
    uniform_colors: list[np.ndarray] = None  # HSV centroids of known uniform colors
    color_tolerance: float = 30.0  # Hue tolerance for uniform matching
    
    def __post_init__(self):
        if self.uniform_colors is None:
            self.uniform_colors = []

    def classify_by_heuristic(
        self,
        total_frames_present: int,
        zones_visited: list[str],
        has_exited: bool,
    ) -> tuple[bool, float]:
        """
        Temporal heuristic for staff detection.
        Returns (is_staff, confidence).
        """
        # Staff typically: present for long time, visit many zones, don't exit
        if has_exited:
            return False, 0.1
        
        if total_frames_present < self.min_presence_frames:
            return False, 0.2
        
        unique_zones = len(set(zones_visited))
        if unique_zones >= self.min_zones_visited:
            # High confidence: long presence + many zones + no exit
            confidence = min(0.95, 0.7 + (unique_zones - self.min_zones_visited) * 0.05)
            return True, confidence
        
        # Moderate: long presence but limited zone coverage
        if total_frames_present > self.min_presence_frames * 2:
            return True, 0.7
        
        return False, 0.3

    def extract_torso_color(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract dominant color from the torso region of a detected person.
        Torso = middle third of the bounding box vertically.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        h = y2 - y1
        
        # Torso: 25% to 60% from top of bbox
        torso_y1 = y1 + int(h * 0.25)
        torso_y2 = y1 + int(h * 0.60)
        
        # Clip to frame bounds
        torso_y1 = max(0, min(torso_y1, frame.shape[0] - 1))
        torso_y2 = max(0, min(torso_y2, frame.shape[0] - 1))
        x1 = max(0, min(x1, frame.shape[1] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        
        if torso_y2 <= torso_y1 or x2 <= x1:
            return None
        
        torso_region = frame[torso_y1:torso_y2, x1:x2]
        if torso_region.size == 0:
            return None
        
        # Convert to HSV and get dominant color
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        # Use mean as a simple dominant color approximation
        mean_color = np.mean(hsv.reshape(-1, 3), axis=0)
        
        return mean_color

    def classify_by_color(self, torso_color: np.ndarray) -> tuple[bool, float]:
        """
        Classify using uniform color matching.
        Returns (is_staff, confidence).
        """
        if not self.uniform_colors or torso_color is None:
            return False, 0.0
        
        for uniform_color in self.uniform_colors:
            # Compare hue primarily (more stable than saturation/value)
            hue_diff = abs(float(torso_color[0]) - float(uniform_color[0]))
            # Hue wraps around at 180 in OpenCV
            hue_diff = min(hue_diff, 180 - hue_diff)
            
            if hue_diff < self.color_tolerance:
                # Also check saturation similarity
                sat_diff = abs(float(torso_color[1]) - float(uniform_color[1]))
                if sat_diff < 50:
                    confidence = max(0.6, 1.0 - (hue_diff / self.color_tolerance))
                    return True, confidence
        
        return False, 0.1

    def learn_uniform_color(self, torso_colors: list[np.ndarray]):
        """
        Learn uniform colors from the first few minutes of footage.
        People present at the very start are likely staff.
        """
        if not torso_colors:
            return
        
        # Simple: average the colors of early-present people
        mean_color = np.mean(torso_colors, axis=0)
        self.uniform_colors.append(mean_color)

    def classify(
        self,
        total_frames_present: int,
        zones_visited: list[str],
        has_exited: bool,
        torso_color: Optional[np.ndarray] = None,
    ) -> tuple[bool, float]:
        """
        Combined classification using both heuristic and color.
        Returns (is_staff, confidence).
        """
        heuristic_result, heuristic_conf = self.classify_by_heuristic(
            total_frames_present, zones_visited, has_exited
        )
        
        color_result, color_conf = False, 0.0
        if torso_color is not None:
            color_result, color_conf = self.classify_by_color(torso_color)
        
        # Combine: either signal can flag staff
        if heuristic_result and heuristic_conf > 0.7:
            return True, heuristic_conf
        
        if color_result and color_conf > 0.7:
            return True, color_conf
        
        # Both signals weak but agreeing
        if heuristic_result and color_result:
            combined_conf = max(heuristic_conf, color_conf)
            return True, combined_conf
        
        return False, max(heuristic_conf, color_conf) * 0.5
