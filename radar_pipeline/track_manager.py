"""
Track Manager Thread - Maintains tracks with Kalman filter state estimation.

Responsibilities:
- Maintain track list with Kalman filter state
- Associate new detections to existing tracks
- Initialize new tracks from unmatched detections
- Classify objects as static vs dynamic
- Maintain static object map
- Publish world state for renderer
"""

import threading
import queue
import time
import numpy as np
from typing import List, Tuple, Optional, Set
from collections import deque

from .data_types import (
    Detection, Track, StaticObject, WorldState, 
    RadarHealth, ObjectClass
)
from .config import PipelineConfig
from .kalman import KalmanFilter
from .association import associate
from .utils import create_occupancy_grid, update_occupancy_grid


class TrackManager(threading.Thread):
    """
    Thread that manages track state and produces WorldState for rendering.
    
    Uses Kalman filtering for state estimation and handles track lifecycle
    including creation, update, classification, and deletion.
    """
    
    def __init__(self,
                 input_queue: queue.Queue,
                 state_queue: queue.Queue,
                 config: PipelineConfig):
        """
        Initialize the track manager thread.
        
        Args:
            input_queue: Queue to consume (frame_num, timestamp, List[Detection]) from
            state_queue: Queue to push WorldState to
            config: Pipeline configuration
        """
        super().__init__(daemon=True, name="TrackManager")
        
        self.input_queue = input_queue
        self.state_queue = state_queue
        self.config = config
        
        self._stop_event = threading.Event()
        
        # Initialize Kalman filter
        self.kalman = KalmanFilter(
            dt=config.frame_period,
            process_noise=config.process_noise,
            measurement_noise=config.measurement_noise
        )
        
        # Track management
        self._tracks: List[Track] = []
        self._static_objects: List[StaticObject] = []
        self._next_track_id = 1
        self._next_static_id = 1
        
        # Occupancy grid
        if config.grid_enabled:
            self._occupancy_grid = create_occupancy_grid(
                config.grid_size,
                config.grid_resolution
            )
        else:
            self._occupancy_grid = None
        
        # Timing
        self._last_update_time = time.time()
        self._frame_count = 0
        
        # FPS tracking
        self._fps_times = deque(maxlen=30)
        
    def run(self):
        """Main thread loop - processes detections and updates tracks."""
        print("[TrackManager] Started")
        
        while not self._stop_event.is_set():
            try:
                # Get detections with timeout
                frame_num, timestamp, detections = self.input_queue.get(timeout=0.1)
                
                # Calculate dt since last update
                current_time = time.time()
                dt = current_time - self._last_update_time
                self._last_update_time = current_time
                
                # Update FPS tracking
                self._fps_times.append(current_time)
                
                # Process the frame
                self._process_frame(frame_num, timestamp, detections, dt)
                
                # Create and publish world state
                world_state = self._create_world_state(frame_num, timestamp)
                
                try:
                    # Put with non-blocking to avoid blocking the track manager
                    self.state_queue.put_nowait(world_state)
                except queue.Full:
                    # If queue is full, drain old states and add new
                    try:
                        while True:
                            self.state_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.state_queue.put_nowait(world_state)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TrackManager] Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("[TrackManager] Stopping...")
    
    def _process_frame(self, 
                       frame_num: int,
                       timestamp: float,
                       detections: List[Detection],
                       dt: float):
        """
        Process a frame of detections.
        
        Args:
            frame_num: Frame number
            timestamp: Frame timestamp
            detections: List of Detection objects
            dt: Time since last update
        """
        self._frame_count = frame_num
        
        # 1. Predict all tracks forward in time
        for track in self._tracks:
            self._predict_track(track, dt)
        
        # 2. Associate detections to tracks
        matches, unmatched_tracks, unmatched_detections = associate(
            self._tracks,
            detections,
            self.kalman,
            gate_threshold=self.config.association_gate,
            use_hungarian=True
        )
        
        # 3. Update matched tracks
        for track_idx, det_idx in matches:
            self._update_track(self._tracks[track_idx], detections[det_idx], timestamp)
        
        # 4. Handle unmatched tracks (increment miss count, decay confidence, but COAST position)
        for track_idx in unmatched_tracks:
            track = self._tracks[track_idx]
            track.misses += 1
            track.hits = 0  # Reset consecutive hits
            
            # Decay confidence slowly (allows coasting through brief dropouts)
            track.confidence = max(0.0, track.confidence - self.config.confidence_decay)
            
            # Add coasted position to history so trail continues during misses
            if track.misses <= 5:  # Only add to trail for first few misses
                track.history.append((track.x, track.y, track.z, timestamp))
        
        # 5. Initialize new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = self._create_track(detections[det_idx], timestamp)
            self._tracks.append(new_track)
        
        # Debug output
        if self.config.debug_tracking and frame_num % 10 == 0:  # Every 10 frames
            confirmed = [t for t in self._tracks if t.confidence >= self.config.min_confidence]
            unconfirmed = [t for t in self._tracks if t.confidence < self.config.min_confidence]
            coasting = [t for t in self._tracks if t.misses > 0 and t.confidence >= self.config.min_confidence]
            print(f"[Track] Frame {frame_num}: "
                  f"Raw={len(detections)} | "
                  f"Confirmed={len(confirmed)} | "
                  f"Unconfirmed={len(unconfirmed)} | "
                  f"Coasting={len(coasting)} | "
                  f"Matches={len(matches)}")
        
        # 6. Classify tracks (static vs dynamic)
        self._classify_tracks()
        
        # 7. Promote static tracks to static objects
        self._promote_static_tracks(timestamp)
        
        # 8. Delete stale tracks
        self._prune_tracks()
        
        # 9. Update occupancy grid
        if self._occupancy_grid is not None:
            for det in detections:
                update_occupancy_grid(
                    self._occupancy_grid,
                    det.x, det.y,
                    self.config.grid_resolution
                )
    
    def _predict_track(self, track: Track, dt: float):
        """Apply Kalman prediction step to a track."""
        track.state, track.covariance = self.kalman.predict(
            track.state, 
            track.covariance,
            dt
        )
        
        # Update position from state
        track.x = track.state[0]
        track.y = track.state[1]
        track.z = track.state[2]
        track.vx = track.state[3]
        track.vy = track.state[4]
        track.vz = track.state[5]
    
    def _update_track(self, track: Track, detection: Detection, timestamp: float):
        """Apply Kalman update step to a track."""
        measurement = np.array([detection.x, detection.y, detection.z])
        
        track.state, track.covariance = self.kalman.update(
            track.state,
            track.covariance,
            measurement
        )
        
        # Update track properties from state
        track.x = track.state[0]
        track.y = track.state[1]
        track.z = track.state[2]
        track.vx = track.state[3]
        track.vy = track.state[4]
        track.vz = track.state[5]
        
        track.snr = detection.snr
        track.last_update = timestamp
        track.hits += 1
        track.misses = 0
        
        # Update confidence
        track.confidence = min(1.0, track.confidence + self.config.confidence_increment)
        
        # Add to history (for trail rendering)
        track.history.append((track.x, track.y, track.z, timestamp))
        
        # Trim history to configured length
        while len(track.history) > self.config.trail_length:
            track.history.pop(0)
    
    def _create_track(self, detection: Detection, timestamp: float) -> Track:
        """Create a new track from a detection."""
        state, covariance = self.kalman.initialize_state(
            detection.x,
            detection.y,
            detection.z
        )
        
        track = Track(
            track_id=self._next_track_id,
            x=detection.x,
            y=detection.y,
            z=detection.z,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            snr=detection.snr,
            classification=ObjectClass.UNKNOWN,
            confidence=self.config.initial_confidence,
            hits=1,
            misses=0,
            history=[(detection.x, detection.y, detection.z, timestamp)],
            last_update=timestamp,
            state=state,
            covariance=covariance,
            low_velocity_frames=0
        )
        
        self._next_track_id += 1
        return track
    
    def _classify_tracks(self):
        """Classify tracks as static or dynamic based on velocity."""
        velocity_threshold = self.config.static_velocity_threshold
        frames_required = self.config.static_frames_required
        
        for track in self._tracks:
            velocity = track.velocity_magnitude
            
            if velocity < velocity_threshold:
                track.low_velocity_frames += 1
                
                if track.low_velocity_frames >= frames_required:
                    track.classification = ObjectClass.STATIC
            else:
                track.low_velocity_frames = 0
                track.classification = ObjectClass.DYNAMIC
    
    def _promote_static_tracks(self, timestamp: float):
        """Promote confirmed static tracks to static objects."""
        tracks_to_remove = []
        
        for track in self._tracks:
            if (track.classification == ObjectClass.STATIC and 
                track.confidence > 0.8 and
                track.hits > self.config.static_frames_required * 2):
                
                # Check if there's already a nearby static object
                is_duplicate = False
                for static in self._static_objects:
                    dist = np.sqrt(
                        (track.x - static.x)**2 +
                        (track.y - static.y)**2 +
                        (track.z - static.z)**2
                    )
                    if dist < 0.3:  # Within 30cm of existing static
                        # Update existing static object
                        static.detections += 1
                        static.last_seen = timestamp
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Create new static object
                    static = StaticObject(
                        x=track.x,
                        y=track.y,
                        z=track.z,
                        radius=0.1,  # Default size estimate
                        detections=track.hits,
                        first_seen=track.history[0][3] if track.history else timestamp,
                        last_seen=timestamp,
                        object_id=self._next_static_id
                    )
                    self._static_objects.append(static)
                    self._next_static_id += 1
                
                # Remove the track (it's now a static object)
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self._tracks.remove(track)
    
    def _prune_tracks(self):
        """Remove stale tracks based on miss count and confidence."""
        max_misses = self.config.max_misses
        min_confidence = self.config.min_confidence
        
        self._tracks = [
            track for track in self._tracks
            if not self._should_delete_track(track, max_misses, min_confidence)
        ]
    
    def _should_delete_track(self, track: Track, max_misses: int, min_confidence: float) -> bool:
        """Check if a track should be deleted."""
        # Delete if too many consecutive misses
        if track.misses > max_misses:
            return True
        
        # Delete if low confidence and missing detections
        if track.confidence < min_confidence and track.misses > 3:
            return True
        
        return False
    
    def _create_world_state(self, frame_num: int, timestamp: float) -> WorldState:
        """Create a WorldState snapshot for the renderer."""
        # Calculate FPS
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            fps = len(self._fps_times) / elapsed if elapsed > 0 else 0
        else:
            fps = 0
        
        # Filter tracks by minimum confidence for display
        display_tracks = [
            track for track in self._tracks
            if track.confidence >= self.config.min_confidence
        ]
        
        # Create health metrics
        health = RadarHealth(
            fps=fps,
            dropped_frames=0,  # Populated by serial reader
            bytes_per_sec=0,
            buffer_size=0,
            total_frames=frame_num,
            active_tracks=len(display_tracks),
            static_objects=len(self._static_objects)
        )
        
        # Copy occupancy grid if enabled
        occupancy_grid = None
        if self._occupancy_grid is not None:
            occupancy_grid = self._occupancy_grid.copy()
        
        return WorldState(
            timestamp=timestamp,
            frame_number=frame_num,
            tracks=display_tracks.copy(),
            static_objects=self._static_objects.copy(),
            occupancy_grid=occupancy_grid,
            radar_health=health
        )
    
    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()
    
    def clear_static_objects(self):
        """Clear all static objects."""
        self._static_objects = []
    
    def get_stats(self) -> dict:
        """Get track manager statistics."""
        return {
            'active_tracks': len(self._tracks),
            'static_objects': len(self._static_objects),
            'frame_count': self._frame_count,
        }

