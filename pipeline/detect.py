"""
Main detection + tracking pipeline.

Processes CCTV clips → detects people → tracks movement → emits structured events.

Usage:
    python detect.py --data-dir /data --store-id STORE_BLR_002 --camera-id CAM_ENTRY_01 --video /data/clips/store_blr_002_entry.mp4
    python detect.py --data-dir /data --process-all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import PipelineConfig, StoreConfig
from tracker import ByteTracker, Detection, Track
from zones import ZoneDetector
from staff import StaffClassifier
from emit import EventEmitter


class DetectionPipeline:
    """
    End-to-end detection pipeline for a single camera feed.
    
    Flow:
    1. Read frame
    2. Run YOLOv8 person detection
    3. Update ByteTracker
    4. Check virtual line crossing (entry/exit)
    5. Zone polygon intersection
    6. Staff classification
    7. Emit structured events
    """

    def __init__(
        self,
        config: PipelineConfig,
        store_id: str,
        camera_id: str,
        video_path: str,
        video_start_time: Optional[datetime] = None,
    ):
        self.config = config
        self.store_id = store_id
        self.camera_id = camera_id
        self.video_path = video_path
        self.video_start_time = video_start_time or datetime.now(timezone.utc)

        # Initialize components
        print(f"[INIT] Loading model: {config.model_path}")
        self.model = YOLO(config.model_path)
        
        self.tracker = ByteTracker(
            high_thresh=0.6,
            low_thresh=config.confidence_threshold,
            track_buffer=config.track_buffer,
        )
        
        self.zone_detector = ZoneDetector(
            config.store_layout_path, store_id, camera_id
        )
        
        self.staff_classifier = StaffClassifier()
        
        self.emitter = EventEmitter(config, store_id)
        
        # State
        self.frame_count = 0
        self.fps = 15.0  # Default; updated from video metadata
        self.frame_height = 1080
        self.frame_width = 1920
        
        # Virtual entry/exit line (horizontal line at configured Y position)
        self.entry_line_y = 0.0  # Set after reading first frame
        
        # Track state for entry/exit detection
        self.track_last_y: dict[str, float] = {}  # track_id -> last Y center
        self.exited_tracks: set[str] = set()  # tracks that have exited
        
        # Dwell tracking: track_id -> {zone_id: enter_frame, last_dwell_emit_frame}
        self.dwell_state: dict[str, dict] = {}
        
        # Re-entry tracking
        self.exited_appearances: list[tuple[str, np.ndarray, int]] = []  # (visitor_id, appearance, exit_frame)
        
        # Queue depth
        self.current_queue_depth = 0
        self.tracks_in_billing: set[str] = set()

    def process(self):
        """Process the entire video file."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {self.video_path}")
            sys.exit(1)

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set entry line based on camera type
        if "ENTRY" in self.camera_id.upper():
            self.entry_line_y = self.frame_height * self.config.entry_line_y
        else:
            self.entry_line_y = self.frame_height * 0.5  # Default mid-frame

        print(f"[START] Processing {self.video_path}")
        print(f"  Store: {self.store_id} | Camera: {self.camera_id}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"  Total frames: {total_frames} (~{total_frames/self.fps:.0f}s)")
        print(f"  Entry line Y: {self.entry_line_y:.0f}px")
        print(f"  Frame skip: {self.config.frame_skip}")

        start_time = time.time()
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Skip frames for performance
            if self.frame_count % self.config.frame_skip != 0:
                continue

            processed_frames += 1
            self._process_frame(frame)

            # Progress logging
            if processed_frames % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = processed_frames / elapsed
                progress = self.frame_count / total_frames * 100
                print(f"  [{progress:.1f}%] Frame {self.frame_count}/{total_frames} "
                      f"| {fps_actual:.1f} fps | Tracks: {len(self.tracker.active_tracks)}")

        cap.release()
        
        # Flush remaining events
        self.emitter.flush()
        
        elapsed = time.time() - start_time
        print(f"\n[DONE] Processed {processed_frames} frames in {elapsed:.1f}s "
              f"({processed_frames/elapsed:.1f} fps)")
        print(f"  Total tracks: {len(self.tracker.get_all_tracks())}")
        stats = self.emitter.get_stats()
        print(f"  Events emitted: visitors={stats['visitors_tracked']}")

    def _process_frame(self, frame: np.ndarray):
        """Process a single frame through the full pipeline."""
        
        # 1. Detect persons
        detections = self._detect_persons(frame)
        
        # 2. Update tracker
        active_tracks = self.tracker.update(detections)
        
        # 3. For each active track: check crossings, zones, staff
        for track in active_tracks:
            if track.last_detection is None:
                continue
            
            bbox = track.last_detection.bbox
            center_y = (bbox[1] + bbox[3]) / 2
            confidence = track.last_detection.confidence
            
            # ─── Entry/Exit detection (virtual line crossing) ─────────────
            if "ENTRY" in self.camera_id.upper():
                self._check_line_crossing(track, center_y, confidence, frame)
            
            # ─── Zone detection ──────────────────────────────────────────
            current_zone = self.zone_detector.detect_zone(bbox)
            self._handle_zone_transition(track, current_zone, confidence, frame)
            
            # ─── Staff classification ────────────────────────────────────
            if track.total_frames_present % 450 == 0:  # Check every ~30s
                torso_color = self.staff_classifier.extract_torso_color(frame, bbox)
                is_staff, staff_conf = self.staff_classifier.classify(
                    track.total_frames_present,
                    track.zones_visited,
                    track.track_id in self.exited_tracks,
                    torso_color,
                )
                track.is_staff = is_staff

            # Update last Y position
            self.track_last_y[track.track_id] = center_y

    def _detect_persons(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8 inference and return person detections."""
        results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.config.confidence_threshold:
                    continue
                
                xyxy = box.xyxy[0].cpu().numpy()
                det = Detection(
                    bbox=xyxy,
                    confidence=conf,
                    class_id=0,
                    frame_idx=self.frame_count,
                )
                detections.append(det)
        
        return detections

    def _check_line_crossing(self, track: Track, center_y: float, confidence: float, frame: np.ndarray):
        """Detect entry/exit by virtual line crossing."""
        prev_y = self.track_last_y.get(track.track_id)
        if prev_y is None:
            return

        line_y = self.entry_line_y
        crossed = False
        direction = None

        # Crossing from above to below = ENTRY (person walking into store)
        if prev_y < line_y and center_y >= line_y:
            crossed = True
            direction = "entry"
        # Crossing from below to above = EXIT
        elif prev_y >= line_y and center_y < line_y:
            crossed = True
            direction = "exit"

        if not crossed:
            return

        if direction == "entry":
            # Check re-entry
            is_reentry = self._check_reentry(track, frame)
            
            if is_reentry:
                self.emitter.emit_reentry(
                    visitor_id=track.track_id,
                    camera_id=self.camera_id,
                    frame_idx=self.frame_count,
                    fps=self.fps,
                    video_start=self.video_start_time,
                    confidence=confidence,
                    is_staff=track.is_staff,
                )
            else:
                self.emitter.emit_entry(
                    visitor_id=track.track_id,
                    camera_id=self.camera_id,
                    frame_idx=self.frame_count,
                    fps=self.fps,
                    video_start=self.video_start_time,
                    confidence=confidence,
                    is_staff=track.is_staff,
                )
            
            # Remove from exited set if re-entering
            self.exited_tracks.discard(track.track_id)

        elif direction == "exit":
            self.emitter.emit_exit(
                visitor_id=track.track_id,
                camera_id=self.camera_id,
                frame_idx=self.frame_count,
                fps=self.fps,
                video_start=self.video_start_time,
                confidence=confidence,
                is_staff=track.is_staff,
            )
            self.exited_tracks.add(track.track_id)
            
            # Store appearance for re-entry detection
            if track.mean_appearance is not None:
                self.exited_appearances.append(
                    (track.track_id, track.mean_appearance, self.frame_count)
                )
                # Keep last 50 exits
                if len(self.exited_appearances) > 50:
                    self.exited_appearances = self.exited_appearances[-50:]

    def _check_reentry(self, track: Track, frame: np.ndarray) -> bool:
        """Check if this entry is a re-entry of a previously exited person."""
        if not self.exited_appearances:
            return False
        
        if track.mean_appearance is None:
            return False

        # Compare against recently exited appearances
        reentry_window_frames = int(self.config.reentry_window_ms / 1000 * self.fps)
        
        for visitor_id, appearance, exit_frame in reversed(self.exited_appearances):
            # Only check within re-entry window
            if self.frame_count - exit_frame > reentry_window_frames:
                continue
            
            # Cosine similarity on appearance features
            from tracker import cosine_similarity
            sim = cosine_similarity(track.mean_appearance, appearance)
            
            if sim > self.config.reentry_similarity_threshold:
                # This is a re-entry - use the original visitor_id
                # Note: in practice we'd reassign the track_id
                return True
        
        return False

    def _handle_zone_transition(self, track: Track, current_zone: Optional[str], confidence: float, frame: np.ndarray):
        """Handle zone enter/exit/dwell events."""
        prev_zone = track.current_zone
        
        if track.track_id not in self.dwell_state:
            self.dwell_state[track.track_id] = {}
        
        # Zone transition
        if current_zone != prev_zone:
            # Exit previous zone
            if prev_zone is not None:
                enter_frame = self.dwell_state[track.track_id].get(f"{prev_zone}_enter", self.frame_count)
                dwell_ms = int((self.frame_count - enter_frame) / self.fps * 1000)
                
                self.emitter.emit_zone_exit(
                    visitor_id=track.track_id,
                    camera_id=self.camera_id,
                    zone_id=prev_zone,
                    frame_idx=self.frame_count,
                    fps=self.fps,
                    video_start=self.video_start_time,
                    confidence=confidence,
                    dwell_ms=dwell_ms,
                    is_staff=track.is_staff,
                )
                
                # Billing queue tracking
                if prev_zone == "BILLING":
                    self.tracks_in_billing.discard(track.track_id)
                    self.current_queue_depth = len(self.tracks_in_billing)
                    # Check if this is an abandonment (no POS in next 5 min - handled by API)
                    self.emitter.emit_billing_queue_abandon(
                        visitor_id=track.track_id,
                        camera_id=self.camera_id,
                        frame_idx=self.frame_count,
                        fps=self.fps,
                        video_start=self.video_start_time,
                        confidence=confidence,
                        is_staff=track.is_staff,
                    )

            # Enter new zone
            if current_zone is not None:
                self.emitter.emit_zone_enter(
                    visitor_id=track.track_id,
                    camera_id=self.camera_id,
                    zone_id=current_zone,
                    frame_idx=self.frame_count,
                    fps=self.fps,
                    video_start=self.video_start_time,
                    confidence=confidence,
                    is_staff=track.is_staff,
                )
                self.dwell_state[track.track_id][f"{current_zone}_enter"] = self.frame_count
                self.dwell_state[track.track_id][f"{current_zone}_last_dwell"] = self.frame_count
                
                if not track.is_staff:
                    track.zones_visited.append(current_zone)
                
                # Billing queue tracking
                if current_zone == "BILLING":
                    self.tracks_in_billing.add(track.track_id)
                    self.current_queue_depth = len(self.tracks_in_billing)
                    if self.current_queue_depth > 0:
                        self.emitter.emit_billing_queue_join(
                            visitor_id=track.track_id,
                            camera_id=self.camera_id,
                            frame_idx=self.frame_count,
                            fps=self.fps,
                            video_start=self.video_start_time,
                            queue_depth=self.current_queue_depth,
                            confidence=confidence,
                            is_staff=track.is_staff,
                        )

            track.current_zone = current_zone

        # Dwell emission (every 30s of continuous presence in same zone)
        elif current_zone is not None:
            last_dwell_frame = self.dwell_state[track.track_id].get(f"{current_zone}_last_dwell", self.frame_count)
            dwell_interval_frames = int(self.config.dwell_emit_interval_ms / 1000 * self.fps)
            
            if self.frame_count - last_dwell_frame >= dwell_interval_frames:
                enter_frame = self.dwell_state[track.track_id].get(f"{current_zone}_enter", self.frame_count)
                total_dwell_ms = int((self.frame_count - enter_frame) / self.fps * 1000)
                
                self.emitter.emit_zone_dwell(
                    visitor_id=track.track_id,
                    camera_id=self.camera_id,
                    zone_id=current_zone,
                    frame_idx=self.frame_count,
                    fps=self.fps,
                    video_start=self.video_start_time,
                    dwell_ms=total_dwell_ms,
                    confidence=confidence,
                    is_staff=track.is_staff,
                )
                self.dwell_state[track.track_id][f"{current_zone}_last_dwell"] = self.frame_count


def discover_videos(data_dir: str) -> list[dict]:
    """Discover video files and infer store/camera metadata from filenames."""
    videos = []
    data_path = Path(data_dir)
    
    # Look for video files
    for ext in ["*.mp4", "*.avi", "*.mov"]:
        for video_file in data_path.rglob(ext):
            # Try to infer store_id and camera_id from path/filename
            name = video_file.stem.lower()
            
            # Guess store_id
            store_id = "STORE_UNKNOWN_001"
            for part in str(video_file).split("/"):
                if "store" in part.lower():
                    store_id = part.upper().replace("-", "_")
                    if not store_id.startswith("STORE_"):
                        store_id = f"STORE_{store_id}"
                    break
            
            # Guess camera_id
            camera_id = "CAM_UNKNOWN_01"
            if "entry" in name or "entrance" in name:
                camera_id = "CAM_ENTRY_01"
            elif "floor" in name or "main" in name:
                camera_id = "CAM_FLOOR_01"
            elif "billing" in name or "checkout" in name:
                camera_id = "CAM_BILLING_01"
            
            videos.append({
                "path": str(video_file),
                "store_id": store_id,
                "camera_id": camera_id,
            })
    
    return videos


def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--data-dir", default="/data", help="Path to data directory")
    parser.add_argument("--store-id", help="Store ID to process")
    parser.add_argument("--camera-id", help="Camera ID")
    parser.add_argument("--video", help="Path to specific video file")
    parser.add_argument("--process-all", action="store_true", help="Process all videos in data dir")
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--confidence", type=float, default=0.3, help="Min confidence threshold")
    
    args = parser.parse_args()
    
    config = PipelineConfig()
    config.api_url = args.api_url
    config.model_path = args.model
    config.frame_skip = args.frame_skip
    config.confidence_threshold = args.confidence
    
    if args.data_dir:
        config.store_layout_path = os.path.join(args.data_dir, "store_layout.json")

    if args.video and args.store_id and args.camera_id:
        # Process single video
        pipeline = DetectionPipeline(
            config=config,
            store_id=args.store_id,
            camera_id=args.camera_id,
            video_path=args.video,
        )
        pipeline.process()
    
    elif args.process_all:
        # Discover and process all videos
        videos = discover_videos(args.data_dir)
        if not videos:
            print(f"[ERROR] No video files found in {args.data_dir}")
            sys.exit(1)
        
        print(f"[DISCOVER] Found {len(videos)} video files:")
        for v in videos:
            print(f"  {v['store_id']} / {v['camera_id']} → {v['path']}")
        
        for video_info in videos:
            print(f"\n{'='*60}")
            pipeline = DetectionPipeline(
                config=config,
                store_id=video_info["store_id"],
                camera_id=video_info["camera_id"],
                video_path=video_info["path"],
            )
            pipeline.process()
    
    else:
        parser.print_help()
        print("\nExample:")
        print("  python detect.py --process-all --data-dir /data")
        print("  python detect.py --video /data/clip.mp4 --store-id STORE_BLR_002 --camera-id CAM_ENTRY_01")


if __name__ == "__main__":
    main()
