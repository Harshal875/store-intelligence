"""ByteTrack-inspired tracker with body appearance features for Re-ID."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    """Single frame detection."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    frame_idx: int
    appearance_feat: Optional[np.ndarray] = None  # body appearance embedding

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class Track:
    """A tracked person across frames."""
    track_id: str = field(default_factory=lambda: f"VIS_{uuid.uuid4().hex[:6]}")
    detections: list[Detection] = field(default_factory=list)
    lost_frames: int = 0
    is_active: bool = True
    
    # State
    predicted_bbox: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None  # [dx, dy] per frame
    
    # Session info
    entered_at: Optional[int] = None  # frame index
    exited_at: Optional[int] = None
    zones_visited: list[str] = field(default_factory=list)
    current_zone: Optional[str] = None
    zone_enter_frame: Optional[int] = None  # frame when entered current zone
    
    # Staff classification
    is_staff: bool = False
    total_frames_present: int = 0
    
    # Re-ID features
    appearance_history: list[np.ndarray] = field(default_factory=list)

    @property
    def last_detection(self) -> Optional[Detection]:
        return self.detections[-1] if self.detections else None

    @property
    def last_bbox(self) -> Optional[np.ndarray]:
        return self.last_detection.bbox if self.last_detection else None

    @property
    def mean_appearance(self) -> Optional[np.ndarray]:
        if not self.appearance_history:
            return None
        return np.mean(self.appearance_history[-10:], axis=0)  # last 10 features

    def update(self, detection: Detection):
        """Update track with new detection."""
        if self.last_bbox is not None:
            center_now = detection.center
            center_prev = ((self.last_bbox[0] + self.last_bbox[2]) / 2,
                          (self.last_bbox[1] + self.last_bbox[3]) / 2)
            self.velocity = np.array([
                center_now[0] - center_prev[0],
                center_now[1] - center_prev[1],
            ])
        
        self.detections.append(detection)
        self.lost_frames = 0
        self.total_frames_present += 1
        
        if detection.appearance_feat is not None:
            self.appearance_history.append(detection.appearance_feat)
            # Keep last 30 features
            if len(self.appearance_history) > 30:
                self.appearance_history = self.appearance_history[-30:]

    def predict(self) -> np.ndarray:
        """Predict next bounding box position using velocity."""
        if self.last_bbox is None:
            return np.zeros(4)
        
        if self.velocity is not None:
            predicted = self.last_bbox.copy()
            predicted[0] += self.velocity[0]
            predicted[1] += self.velocity[1]
            predicted[2] += self.velocity[0]
            predicted[3] += self.velocity[1]
            self.predicted_bbox = predicted
            return predicted
        
        self.predicted_bbox = self.last_bbox.copy()
        return self.predicted_bbox

    def mark_lost(self):
        """Mark track as lost for this frame."""
        self.lost_frames += 1


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(feat1, feat2) / (norm1 * norm2))


class ByteTracker:
    """
    ByteTrack-inspired multi-object tracker.
    
    Two-stage association:
    1. High-confidence detections matched to tracks via IoU
    2. Low-confidence detections matched to remaining tracks
    
    No appearance model in primary association (handles face blur).
    Appearance features used only for Re-ID across camera gaps.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.3,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        
        self.active_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.finished_tracks: list[Track] = []
        self.frame_count: int = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Process one frame of detections. Returns active tracks.
        """
        self.frame_count += 1
        
        if not detections:
            # Age all tracks
            for track in self.active_tracks:
                track.mark_lost()
            
            # Move lost tracks
            still_active = []
            for track in self.active_tracks:
                if track.lost_frames > self.track_buffer:
                    track.is_active = False
                    self.finished_tracks.append(track)
                else:
                    self.lost_tracks.append(track)
                    still_active.append(track)
            self.active_tracks = still_active
            return self.active_tracks

        # Split detections by confidence
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # ─── First association: high-conf detections vs active tracks ─────
        unmatched_tracks = []
        unmatched_dets = []
        
        if self.active_tracks and high_dets:
            cost_matrix = self._compute_iou_cost(self.active_tracks, high_dets)
            matched_indices, unmatched_track_idx, unmatched_det_idx = self._linear_assignment(
                cost_matrix, thresh=self.match_thresh
            )
            
            for t_idx, d_idx in matched_indices:
                self.active_tracks[t_idx].update(high_dets[d_idx])
            
            unmatched_tracks = [self.active_tracks[i] for i in unmatched_track_idx]
            unmatched_dets = [high_dets[i] for i in unmatched_det_idx]
        elif high_dets:
            unmatched_dets = high_dets
        else:
            unmatched_tracks = list(self.active_tracks)

        # ─── Second association: low-conf detections vs unmatched tracks ──
        if unmatched_tracks and low_dets:
            cost_matrix = self._compute_iou_cost(unmatched_tracks, low_dets)
            matched_indices, still_unmatched_t, _ = self._linear_assignment(
                cost_matrix, thresh=self.match_thresh
            )
            
            for t_idx, d_idx in matched_indices:
                unmatched_tracks[t_idx].update(low_dets[d_idx])
            
            # Remaining unmatched tracks become lost
            for i in still_unmatched_t:
                unmatched_tracks[i].mark_lost()
        else:
            for track in unmatched_tracks:
                track.mark_lost()

        # ─── Initialize new tracks from unmatched high-conf detections ────
        for det in unmatched_dets:
            new_track = Track()
            new_track.update(det)
            new_track.entered_at = self.frame_count
            self.active_tracks.append(new_track)

        # ─── Handle lost tracks ──────────────────────────────────────────
        new_active = []
        for track in self.active_tracks:
            if track.lost_frames > self.track_buffer:
                track.is_active = False
                self.finished_tracks.append(track)
            else:
                new_active.append(track)
        self.active_tracks = new_active

        return self.active_tracks

    def _compute_iou_cost(self, tracks: list[Track], detections: list[Detection]) -> np.ndarray:
        """Compute IoU cost matrix between tracks and detections."""
        cost = np.zeros((len(tracks), len(detections)))
        for t_idx, track in enumerate(tracks):
            predicted = track.predict()
            for d_idx, det in enumerate(detections):
                iou = compute_iou(predicted, det.bbox)
                cost[t_idx, d_idx] = 1.0 - iou  # cost = 1 - IoU
        return cost

    def _linear_assignment(
        self, cost_matrix: np.ndarray, thresh: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Solve linear assignment problem."""
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_rows = set(range(cost_matrix.shape[0]))
        unmatched_cols = set(range(cost_matrix.shape[1]))
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1.0 - thresh + 1.0):  # Convert back: iou >= thresh equivalent
                # Actually: cost <= 1 - match_thresh means IoU >= match_thresh
                # But we want to accept more matches, so use a relaxed threshold
                if cost_matrix[r, c] < 1.0:  # Any positive IoU
                    matched.append((r, c))
                    unmatched_rows.discard(r)
                    unmatched_cols.discard(c)
        
        return matched, list(unmatched_rows), list(unmatched_cols)

    def get_all_tracks(self) -> list[Track]:
        """Get all tracks (active + finished)."""
        return self.active_tracks + self.finished_tracks
