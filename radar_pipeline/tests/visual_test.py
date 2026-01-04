#!/usr/bin/env python3
"""
Visual test for the radar pipeline.

Displays synthetic moving targets in the 3D visualizer to validate
that tracking and visualization are working correctly.

Usage:
    python3 radar_pipeline/tests/visual_test.py [scenario]

Scenarios:
    1 - Single target moving forward (default)
    2 - Target moving in a circle
    3 - Two targets crossing paths
    4 - Target with dropout (coasting test)
    5 - Target moving up and down (Z test)
    6 - Random walk (stress test)
"""

import sys
import os
import time
import queue
import threading
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyqtgraph.Qt import QtWidgets, QtGui, QtCore

from radar_pipeline.data_types import Detection, Track, ObjectClass, WorldState, RadarHealth
from radar_pipeline.config import PipelineConfig
from radar_pipeline.kalman import KalmanFilter
from radar_pipeline.render_engine import RenderEngine


class SyntheticTracker:
    """Simplified tracker for synthetic data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.kalman = KalmanFilter(
            dt=config.frame_period,
            process_noise=config.process_noise,
            measurement_noise=config.measurement_noise
        )
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
    
    def process_detections(self, detections: list, timestamp: float) -> WorldState:
        """Process detections and return WorldState."""
        self.frame_count += 1
        dt = self.config.frame_period
        
        # Predict all tracks
        for track in self.tracks:
            track.state, track.covariance = self.kalman.predict(
                track.state, track.covariance, dt
            )
            track.x, track.y, track.z = track.state[0], track.state[1], track.state[2]
            track.vx, track.vy, track.vz = track.state[3], track.state[4], track.state[5]
        
        # Simple nearest-neighbor association
        matched_tracks = set()
        matched_dets = set()
        
        for i, det in enumerate(detections):
            best_track = None
            best_dist = self.config.association_gate
            
            for j, track in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                dist = track.distance_to(det.x, det.y, det.z)
                if dist < best_dist:
                    best_dist = dist
                    best_track = j
            
            if best_track is not None:
                matched_tracks.add(best_track)
                matched_dets.add(i)
                
                # Update track
                track = self.tracks[best_track]
                measurement = np.array([det.x, det.y, det.z])
                track.state, track.covariance = self.kalman.update(
                    track.state, track.covariance, measurement
                )
                track.x, track.y, track.z = track.state[0], track.state[1], track.state[2]
                track.vx, track.vy, track.vz = track.state[3], track.state[4], track.state[5]
                track.hits += 1
                track.misses = 0
                track.confidence = min(1.0, track.confidence + self.config.confidence_increment)
                track.last_update = timestamp
                track.history.append((track.x, track.y, track.z, timestamp))
                if len(track.history) > self.config.trail_length:
                    track.history.pop(0)
        
        # Handle unmatched tracks
        for j, track in enumerate(self.tracks):
            if j not in matched_tracks:
                track.misses += 1
                track.hits = 0
                track.confidence = max(0.0, track.confidence - self.config.confidence_decay)
                if track.misses <= 5:
                    track.history.append((track.x, track.y, track.z, timestamp))
        
        # Create new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                state, cov = self.kalman.initialize_state(det.x, det.y, det.z)
                track = Track(
                    track_id=self.next_id,
                    x=det.x, y=det.y, z=det.z,
                    vx=0.0, vy=0.0, vz=0.0,
                    snr=det.snr,
                    classification=ObjectClass.DYNAMIC,
                    confidence=self.config.initial_confidence,
                    hits=1, misses=0,
                    history=[(det.x, det.y, det.z, timestamp)],
                    last_update=timestamp,
                    state=state, covariance=cov
                )
                self.tracks.append(track)
                self.next_id += 1
        
        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.config.max_misses]
        
        # Mark tracks as confirmed
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.config.min_hits:
                t.confirmed = True
        
        # Filter for display
        display_tracks = [t for t in self.tracks if t.confirmed and t.confidence >= self.config.min_confidence]
        
        health = RadarHealth(
            fps=20.0,
            total_frames=self.frame_count,
            active_tracks=len(display_tracks)
        )
        
        return WorldState(
            timestamp=timestamp,
            frame_number=self.frame_count,
            tracks=display_tracks,
            static_objects=[],
            occupancy_grid=None,
            radar_health=health
        )


class ScenarioGenerator:
    """Generates synthetic radar detections for different test scenarios."""
    
    def __init__(self, scenario: int = 1):
        self.scenario = scenario
        self.start_time = time.time()
    
    def get_detections(self, t: float) -> list:
        """Get detections for current time."""
        if self.scenario == 1:
            return self._single_target_forward(t)
        elif self.scenario == 2:
            return self._circular_motion(t)
        elif self.scenario == 3:
            return self._crossing_targets(t)
        elif self.scenario == 4:
            return self._dropout_test(t)
        elif self.scenario == 5:
            return self._vertical_motion(t)
        elif self.scenario == 6:
            return self._random_walk(t)
        else:
            return self._single_target_forward(t)
    
    def _add_noise(self, x, y, z, noise=0.03):
        """Add measurement noise."""
        return (
            x + np.random.randn() * noise,
            y + np.random.randn() * noise,
            z + np.random.randn() * noise
        )
    
    def _single_target_forward(self, t):
        """Single target moving forward then back."""
        # Move from y=2 to y=6, then back
        period = 8.0  # seconds for full cycle
        phase = (t % period) / period
        
        if phase < 0.5:
            y = 2.0 + 8.0 * phase  # Forward
        else:
            y = 6.0 - 8.0 * (phase - 0.5)  # Back
        
        x, y, z = self._add_noise(0.0, y, 0.5)
        return [Detection(x=x, y=y, z=z, snr=20.0, timestamp=t)]
    
    def _circular_motion(self, t):
        """Target moving in a circle."""
        radius = 2.0
        center_y = 4.0
        omega = 0.5  # rad/s
        
        x = radius * np.sin(omega * t)
        y = center_y + radius * np.cos(omega * t)
        z = 0.5 + 0.3 * np.sin(omega * t * 2)  # slight Z oscillation
        
        x, y, z = self._add_noise(x, y, z)
        return [Detection(x=x, y=y, z=z, snr=20.0, timestamp=t)]
    
    def _crossing_targets(self, t):
        """Two targets crossing paths."""
        # Target 1: moves left to right
        x1 = -2.0 + 0.5 * t
        y1 = 3.0
        z1 = 0.4
        
        # Target 2: moves right to left
        x2 = 2.0 - 0.5 * t
        y2 = 4.0
        z2 = 0.6
        
        detections = []
        
        # Only show if within bounds
        if -3 < x1 < 3:
            x, y, z = self._add_noise(x1, y1, z1)
            detections.append(Detection(x=x, y=y, z=z, snr=20.0, timestamp=t))
        
        if -3 < x2 < 3:
            x, y, z = self._add_noise(x2, y2, z2)
            detections.append(Detection(x=x, y=y, z=z, snr=18.0, timestamp=t))
        
        return detections
    
    def _dropout_test(self, t):
        """Target with periodic dropouts to test coasting."""
        y = 2.0 + 0.5 * t
        
        # Drop out for 0.5s every 2s
        cycle_phase = t % 2.0
        if 1.0 < cycle_phase < 1.5:
            return []  # Dropout!
        
        x, y, z = self._add_noise(0.0, min(y, 6.0), 0.5)
        return [Detection(x=x, y=y, z=z, snr=20.0, timestamp=t)]
    
    def _vertical_motion(self, t):
        """Target moving up and down (Z test)."""
        # Oscillate in Z
        z = 0.5 + 0.8 * np.sin(0.5 * t)
        y = 3.0 + 0.3 * t  # slow forward movement
        
        x, y, z = self._add_noise(0.0, min(y, 5.0), z)
        return [Detection(x=x, y=y, z=z, snr=20.0, timestamp=t)]
    
    def _random_walk(self, t):
        """Random walk stress test."""
        # Use time-based seed for reproducibility within session
        np.random.seed(int(t * 20) % 10000)
        
        # Base position with slow drift
        x = np.sin(t * 0.2) * 1.5
        y = 3.0 + np.cos(t * 0.15) * 1.0
        z = 0.5 + np.sin(t * 0.3) * 0.3
        
        # Add larger random component
        x, y, z = self._add_noise(x, y, z, noise=0.1)
        
        return [Detection(x=x, y=y, z=z, snr=20.0, timestamp=t)]


class VisualTest:
    """Visual test runner with Qt integration."""
    
    def __init__(self, scenario: int = 1):
        self.scenario = scenario
        self.config = PipelineConfig()
        self.state_queue = queue.Queue(maxsize=10)
        self.running = False
        
        self.tracker = SyntheticTracker(self.config)
        self.generator = ScenarioGenerator(scenario)
    
    def _simulation_loop(self):
        """Run simulation in background thread."""
        start_time = time.time()
        frame = 0
        
        while self.running:
            t = time.time() - start_time
            
            # Generate detections
            detections = self.generator.get_detections(t)
            
            # Process through tracker
            world_state = self.tracker.process_detections(detections, t)
            
            # Send to render queue
            try:
                self.state_queue.put_nowait(world_state)
            except queue.Full:
                try:
                    self.state_queue.get_nowait()
                    self.state_queue.put_nowait(world_state)
                except:
                    pass
            
            frame += 1
            time.sleep(0.05)  # 20 Hz
    
    def run(self):
        """Run the visual test."""
        scenario_names = {
            1: "Single target moving forward/back",
            2: "Circular motion",
            3: "Two targets crossing paths",
            4: "Dropout test (coasting)",
            5: "Vertical motion (Z test)",
            6: "Random walk (stress test)",
        }
        
        print("="*60)
        print("VISUAL TEST - Radar Pipeline")
        print("="*60)
        print(f"Scenario {self.scenario}: {scenario_names.get(self.scenario, 'Unknown')}")
        print("="*60)
        print("Controls: Left-drag to rotate, Right-drag to zoom")
        print("Close window to exit")
        print("="*60)
        
        # Create Qt app
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        
        palette = app.palette()
        palette.setColor(palette.ColorRole.Window, QtGui.QColor('#0a0a12'))
        palette.setColor(palette.ColorRole.WindowText, QtGui.QColor('#e0e0e0'))
        app.setPalette(palette)
        
        # Create render engine
        render = RenderEngine(self.state_queue, self.config)
        render.setWindowTitle(f"Visual Test - Scenario {self.scenario}: {scenario_names.get(self.scenario, '')}")
        render.show()
        
        # Start simulation thread
        self.running = True
        sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        sim_thread.start()
        
        # Run Qt event loop
        exit_code = app.exec()
        
        self.running = False
        sim_thread.join(timeout=1.0)
        
        return exit_code


def main():
    parser = argparse.ArgumentParser(description="Visual test for radar pipeline")
    parser.add_argument('scenario', type=int, nargs='?', default=1,
                        help="Test scenario (1-6)")
    args = parser.parse_args()
    
    if args.scenario < 1 or args.scenario > 6:
        print("Invalid scenario. Choose 1-6:")
        print("  1 - Single target moving forward")
        print("  2 - Circular motion")
        print("  3 - Two targets crossing")
        print("  4 - Dropout test (coasting)")
        print("  5 - Vertical motion (Z test)")
        print("  6 - Random walk")
        return 1
    
    test = VisualTest(args.scenario)
    return test.run()


if __name__ == "__main__":
    sys.exit(main())

