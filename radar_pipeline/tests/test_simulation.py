"""
Simulation tests with synthetic moving targets.

These tests validate that the tracking pipeline correctly handles:
- Moving targets
- Track coasting during dropouts
- Multiple simultaneous tracks
- Static vs dynamic classification
"""

import numpy as np
import time
import sys
sys.path.insert(0, '../..')

from radar_pipeline.data_types import Detection, Track, ObjectClass
from radar_pipeline.config import PipelineConfig
from radar_pipeline.kalman import KalmanFilter
from radar_pipeline.association import associate


class SimpleTracker:
    """
    Simplified tracker for testing (same logic as TrackManager but synchronous).
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.kalman = KalmanFilter(
            dt=self.config.frame_period,
            process_noise=self.config.process_noise,
            measurement_noise=self.config.measurement_noise
        )
        self.tracks = []
        self.next_id = 1
    
    def process_detections(self, detections: list, timestamp: float) -> list:
        """Process a list of detections and return confirmed tracks."""
        dt = self.config.frame_period
        
        # 1. Predict all tracks
        for track in self.tracks:
            track.state, track.covariance = self.kalman.predict(
                track.state, track.covariance, dt
            )
            track.x, track.y, track.z = track.state[0], track.state[1], track.state[2]
            track.vx, track.vy, track.vz = track.state[3], track.state[4], track.state[5]
        
        # 2. Associate
        matches, unmatched_t, unmatched_d = associate(
            self.tracks, detections, self.kalman,
            gate_threshold=self.config.association_gate
        )
        
        # 3. Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]
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
        
        # 4. Handle unmatched tracks
        for track_idx in unmatched_t:
            self.tracks[track_idx].misses += 1
            self.tracks[track_idx].hits = 0
            self.tracks[track_idx].confidence = max(
                0.0, 
                self.tracks[track_idx].confidence - self.config.confidence_decay
            )
        
        # 5. Create new tracks
        for det_idx in unmatched_d:
            det = detections[det_idx]
            state, cov = self.kalman.initialize_state(det.x, det.y, det.z)
            track = Track(
                track_id=self.next_id,
                x=det.x, y=det.y, z=det.z,
                vx=0.0, vy=0.0, vz=0.0,
                snr=det.snr,
                classification=ObjectClass.UNKNOWN,
                confidence=self.config.initial_confidence,
                hits=1, misses=0,
                history=[(det.x, det.y, det.z, timestamp)],
                last_update=timestamp,
                state=state, covariance=cov
            )
            self.tracks.append(track)
            self.next_id += 1
        
        # 6. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.config.max_misses]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.confidence >= self.config.min_confidence]


def test_single_moving_target():
    """Test tracking a single target moving in a straight line."""
    tracker = SimpleTracker()
    
    # Simulate target moving from (0, 2, 0.5) at 1 m/s in Y direction
    start_x, start_y, start_z = 0.0, 2.0, 0.5
    velocity_y = 1.0  # m/s
    dt = 0.05  # 20 Hz
    
    confirmed_tracks = []
    
    for frame in range(40):  # 2 seconds
        t = frame * dt
        true_y = start_y + velocity_y * t
        
        # Add small noise
        det = Detection(
            x=start_x + np.random.randn() * 0.02,
            y=true_y + np.random.randn() * 0.02,
            z=start_z + np.random.randn() * 0.02,
            snr=15.0,
            timestamp=t
        )
        
        confirmed_tracks = tracker.process_detections([det], t)
    
    # Should have exactly 1 confirmed track
    assert len(confirmed_tracks) == 1, f"Expected 1 track, got {len(confirmed_tracks)}"
    
    # Track should be near final position
    track = confirmed_tracks[0]
    expected_y = start_y + velocity_y * (39 * dt)
    assert abs(track.y - expected_y) < 0.2, f"Track Y {track.y} too far from expected {expected_y}"
    
    # Velocity should be estimated correctly
    assert abs(track.vy - velocity_y) < 0.3, f"Velocity {track.vy} too far from {velocity_y}"
    
    print(f"✓ Single moving target test passed")
    print(f"  Final position: ({track.x:.2f}, {track.y:.2f}, {track.z:.2f})")
    print(f"  Estimated velocity: ({track.vx:.2f}, {track.vy:.2f}, {track.vz:.2f}) m/s")


def test_target_with_dropout():
    """Test that tracker coasts through missed detections."""
    tracker = SimpleTracker()
    
    # Target moving at 1 m/s in Y
    velocity_y = 1.0
    dt = 0.05
    
    confirmed_before_dropout = None
    confirmed_during_dropout = None
    confirmed_after_dropout = None
    
    for frame in range(60):
        t = frame * dt
        true_y = 2.0 + velocity_y * t
        
        # Dropout from frames 20-30 (0.5 seconds)
        if 20 <= frame < 30:
            detections = []  # No detection
        else:
            detections = [Detection(
                x=0.0 + np.random.randn() * 0.02,
                y=true_y + np.random.randn() * 0.02,
                z=0.5 + np.random.randn() * 0.02,
                snr=15.0, timestamp=t
            )]
        
        confirmed = tracker.process_detections(detections, t)
        
        if frame == 19:
            confirmed_before_dropout = len(confirmed)
        if frame == 25:
            confirmed_during_dropout = len(confirmed)
        if frame == 40:
            confirmed_after_dropout = len(confirmed)
    
    # Should maintain track through 10-frame dropout
    assert confirmed_before_dropout == 1, f"Should have track before dropout, got {confirmed_before_dropout}"
    assert confirmed_during_dropout >= 1, "Should coast during dropout"
    assert confirmed_after_dropout == 1, "Should recover after dropout"
    
    # Should still have 1 track at the end
    assert len(confirmed) == 1, f"Expected 1 track after recovery, got {len(confirmed)}"
    
    print("✓ Target with dropout test passed")
    print(f"  Tracks before/during/after dropout: {confirmed_before_dropout}/{confirmed_during_dropout}/{confirmed_after_dropout}")


def test_multiple_targets():
    """Test tracking multiple simultaneous targets."""
    tracker = SimpleTracker()
    
    # Two targets moving in parallel
    targets = [
        {'x': -1.0, 'y': 2.0, 'z': 0.5, 'vx': 0.0, 'vy': 1.0, 'vz': 0.0},
        {'x': 1.0, 'y': 3.0, 'z': 0.3, 'vx': 0.0, 'vy': 0.5, 'vz': 0.0},
    ]
    
    dt = 0.05
    
    for frame in range(40):
        t = frame * dt
        
        detections = []
        for target in targets:
            det = Detection(
                x=target['x'] + target['vx'] * t + np.random.randn() * 0.02,
                y=target['y'] + target['vy'] * t + np.random.randn() * 0.02,
                z=target['z'] + target['vz'] * t + np.random.randn() * 0.02,
                snr=15.0, timestamp=t
            )
            detections.append(det)
        
        confirmed = tracker.process_detections(detections, t)
    
    # Should have 2 tracks
    assert len(confirmed) == 2, f"Expected 2 tracks, got {len(confirmed)}"
    
    # Tracks should be separated
    dist = confirmed[0].distance_to(confirmed[1].x, confirmed[1].y, confirmed[1].z)
    assert dist > 1.0, f"Tracks too close: {dist}m"
    
    print("✓ Multiple targets test passed")
    print(f"  Track 1: ({confirmed[0].x:.2f}, {confirmed[0].y:.2f}, {confirmed[0].z:.2f})")
    print(f"  Track 2: ({confirmed[1].x:.2f}, {confirmed[1].y:.2f}, {confirmed[1].z:.2f})")


def test_z_tracking():
    """Test that Z dimension is tracked correctly for moving target."""
    tracker = SimpleTracker()
    
    # Target moving UP (increasing Z)
    start_z = 0.0
    velocity_z = 0.5  # m/s upward
    dt = 0.05
    
    for frame in range(40):
        t = frame * dt
        true_z = start_z + velocity_z * t
        
        det = Detection(
            x=0.0 + np.random.randn() * 0.02,
            y=2.0 + np.random.randn() * 0.02,
            z=true_z + np.random.randn() * 0.02,
            snr=15.0, timestamp=t
        )
        
        confirmed = tracker.process_detections([det], t)
    
    assert len(confirmed) == 1, f"Expected 1 track, got {len(confirmed)}"
    
    track = confirmed[0]
    expected_z = start_z + velocity_z * (39 * dt)
    
    assert abs(track.z - expected_z) < 0.2, f"Track Z {track.z} too far from expected {expected_z}"
    assert abs(track.vz - velocity_z) < 0.2, f"Z velocity {track.vz} too far from {velocity_z}"
    
    print("✓ Z tracking test passed")
    print(f"  Final Z: {track.z:.2f}m (expected ~{expected_z:.2f}m)")
    print(f"  Z velocity: {track.vz:.2f} m/s (expected ~{velocity_z:.2f} m/s)")


def test_track_confirmation_timing():
    """Test how quickly tracks become confirmed."""
    config = PipelineConfig()
    tracker = SimpleTracker(config)

    confirmation_frame = None

    for frame in range(20):
        t = frame * 0.05
        det = Detection(x=0.0, y=2.0, z=0.5, snr=15.0, timestamp=t)
        confirmed = tracker.process_detections([det], t)

        if confirmed and confirmation_frame is None:
            confirmation_frame = frame

    # Calculate expected confirmation frame based on config
    # Track starts with initial_confidence, needs to reach min_confidence
    # Each hit adds confidence_increment
    # Frame 0: create track with initial_confidence (1 hit)
    # Frame N: confidence = initial + N * increment
    # Need: initial + N * increment >= min_confidence
    # N >= (min_confidence - initial) / increment
    initial = config.initial_confidence
    min_conf = config.min_confidence
    increment = config.confidence_increment

    if initial >= min_conf:
        expected_frame = 0
    else:
        hits_needed = int(np.ceil((min_conf - initial) / increment))
        expected_frame = hits_needed  # Frame 0 creates, so N more hits = frame N

    assert confirmation_frame is not None, "Track never confirmed"
    assert confirmation_frame <= expected_frame + 1, \
        f"Track confirmed at frame {confirmation_frame}, expected <= {expected_frame + 1}"

    print(f"✓ Track confirmation timing test passed (confirmed at frame {confirmation_frame})")


def run_all_tests():
    """Run all simulation tests."""
    print("\n" + "="*50)
    print("SIMULATION TESTS")
    print("="*50 + "\n")
    
    test_single_moving_target()
    test_target_with_dropout()
    test_multiple_targets()
    test_z_tracking()
    test_track_confirmation_timing()
    
    print("\n" + "="*50)
    print("ALL SIMULATION TESTS PASSED ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()

