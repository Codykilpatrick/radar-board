#!/usr/bin/env python3
"""
Visual test for the radar pipeline.

Displays synthetic moving targets in the 3D visualizer to validate
that tracking and visualization are working correctly.

Usage:
    python3 radar_pipeline/tests/visual_test.py [scenario]

Scenarios:
    1 - forward    - Single target moving forward/back
    2 - circular   - Target moving in a circle
    3 - crossing   - Two targets crossing paths
    4 - dropout    - Target with dropouts (coasting test)
    5 - vertical   - Target moving up and down (Z test)
    6 - random     - Random walk (stress test)
    7 - multi      - Multiple targets at different ranges
    8 - clutter    - Static clutter with moving target
    9 - approaching - Target approaching the radar

Or use scenario name directly:
    python3 radar_pipeline/tests/visual_test.py crossing
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
from radar_pipeline.scenarios import SCENARIOS, list_scenarios


# Map numeric scenarios to names for backwards compatibility
SCENARIO_MAP = {
    1: 'forward',
    2: 'circular',
    3: 'crossing',
    4: 'dropout',
    5: 'vertical',
    6: 'random',
    7: 'multi',
    8: 'clutter',
    9: 'approaching',
}


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


class VisualTest:
    """Visual test runner with Qt integration."""

    def __init__(self, scenario_name: str = 'forward'):
        self.scenario_name = scenario_name
        self.config = PipelineConfig()
        self.state_queue = queue.Queue(maxsize=10)
        self.running = False

        self.tracker = SyntheticTracker(self.config)
        # Use the scenario from the scenarios module
        self.generator = SCENARIOS[scenario_name]()

    def _simulation_loop(self):
        """Run simulation in background thread."""
        start_time = time.time()
        frame = 0

        while self.running:
            t = time.time() - start_time

            # Generate detections using the scenario
            detections = self.generator.generate(t)

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
        print("="*60)
        print("VISUAL TEST - Radar Pipeline")
        print("="*60)
        print(f"Scenario: {self.scenario_name} - {self.generator.description}")
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
        render.setWindowTitle(f"Visual Test - {self.scenario_name}: {self.generator.description}")
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
    parser = argparse.ArgumentParser(
        description="Visual test for radar pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 1          # Scenario 1 (forward)
  %(prog)s crossing   # Scenario by name
  %(prog)s --list     # List all scenarios
        """
    )
    parser.add_argument('scenario', nargs='?', default='1',
                        help="Test scenario (number 1-9 or name)")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List available scenarios and exit")
    args = parser.parse_args()

    # Handle --list
    if args.list:
        print("Available scenarios:")
        print("-" * 60)
        for i, (name, desc) in enumerate(list_scenarios().items(), 1):
            print(f"  {i:2} / {name:12} - {desc}")
        print("-" * 60)
        return 0

    # Parse scenario (number or name)
    scenario_name = args.scenario
    if scenario_name.isdigit():
        num = int(scenario_name)
        if num in SCENARIO_MAP:
            scenario_name = SCENARIO_MAP[num]
        else:
            print(f"Invalid scenario number: {num}")
            print("Use --list to see available scenarios")
            return 1
    elif scenario_name not in SCENARIOS:
        print(f"Unknown scenario: {scenario_name}")
        print("Use --list to see available scenarios")
        return 1

    test = VisualTest(scenario_name)
    return test.run()


if __name__ == "__main__":
    sys.exit(main())

