# PROMPT: "Generate pytest tests for the detection pipeline's tracker module:
# ByteTracker basic tracking, track ID persistence across frames,
# IoU computation correctness, lost track handling, and new track initialization."
#
# CHANGES MADE: Added tests for group detection (multiple tracks from simultaneous detections),
# and track velocity computation. Made tests unit-level (no video required).

"""Tests for pipeline tracker module."""

import numpy as np
import pytest

import sys
sys.path.insert(0, "pipeline")

from pipeline.tracker import ByteTracker, Detection, Track, compute_iou, cosine_similarity


class TestIoU:
    """Test IoU computation."""

    def test_iou_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        box = np.array([10, 10, 50, 50])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])
        assert compute_iou(box1, box2) == 0.0

    def test_iou_partial_overlap(self):
        """Partially overlapping boxes."""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        assert compute_iou(box1, box2) == pytest.approx(25.0 / 175.0, abs=0.01)


class TestCosineSimilarity:
    """Test cosine similarity."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=0.001)

    def test_zero_vector(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([0.0, 0.0])
        assert cosine_similarity(v1, v2) == 0.0


class TestByteTracker:
    """Test ByteTracker tracking logic."""

    def test_single_detection_creates_track(self):
        """A single detection should create one track."""
        tracker = ByteTracker()
        det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=0,
            frame_idx=1,
        )
        tracks = tracker.update([det])
        assert len(tracks) == 1
        assert tracks[0].is_active

    def test_track_persistence(self):
        """Track should persist across frames with nearby detections."""
        tracker = ByteTracker(track_buffer=10)
        
        # Frame 1
        det1 = Detection(bbox=np.array([100, 100, 200, 200]), confidence=0.9, class_id=0, frame_idx=1)
        tracks1 = tracker.update([det1])
        track_id = tracks1[0].track_id
        
        # Frame 2 - slightly moved
        det2 = Detection(bbox=np.array([105, 105, 205, 205]), confidence=0.9, class_id=0, frame_idx=2)
        tracks2 = tracker.update([det2])
        
        assert len(tracks2) == 1
        assert tracks2[0].track_id == track_id  # Same track ID

    def test_multiple_simultaneous_detections(self):
        """Multiple people in same frame should create separate tracks (group handling)."""
        tracker = ByteTracker()
        
        detections = [
            Detection(bbox=np.array([100, 100, 150, 300]), confidence=0.9, class_id=0, frame_idx=1),
            Detection(bbox=np.array([200, 100, 250, 300]), confidence=0.9, class_id=0, frame_idx=1),
            Detection(bbox=np.array([300, 100, 350, 300]), confidence=0.85, class_id=0, frame_idx=1),
        ]
        tracks = tracker.update(detections)
        
        assert len(tracks) == 3  # 3 separate people, 3 separate tracks

    def test_lost_track_removal(self):
        """Track lost for more than track_buffer frames should be removed."""
        tracker = ByteTracker(track_buffer=3)
        
        # Create track
        det = Detection(bbox=np.array([100, 100, 200, 200]), confidence=0.9, class_id=0, frame_idx=1)
        tracker.update([det])
        
        # Empty frames
        for i in range(5):
            tracker.update([])
        
        assert len(tracker.active_tracks) == 0
        assert len(tracker.finished_tracks) == 1

    def test_empty_frame_handling(self):
        """Empty frames should not crash."""
        tracker = ByteTracker()
        tracks = tracker.update([])
        assert len(tracks) == 0

    def test_track_velocity(self):
        """Track should compute velocity from consecutive detections."""
        tracker = ByteTracker()
        
        det1 = Detection(bbox=np.array([100, 100, 200, 200]), confidence=0.9, class_id=0, frame_idx=1)
        tracker.update([det1])
        
        det2 = Detection(bbox=np.array([110, 120, 210, 220]), confidence=0.9, class_id=0, frame_idx=2)
        tracks = tracker.update([det2])
        
        # Velocity should be [10, 20] (center moved by 10px right, 20px down)
        assert tracks[0].velocity is not None
        assert tracks[0].velocity[0] == pytest.approx(10.0, abs=1)
        assert tracks[0].velocity[1] == pytest.approx(20.0, abs=1)
