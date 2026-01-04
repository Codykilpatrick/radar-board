"""
Tests for track lifecycle and management.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '../..')

from radar_pipeline.data_types import Detection, Track, ObjectClass, WorldState
from radar_pipeline.config import PipelineConfig
from radar_pipeline.kalman import KalmanFilter


def create_detection(x: float, y: float, z: float, t: float = 0.0) -> Detection:
    """Helper to create a detection."""
    return Detection(x=x, y=y, z=z, snr=15.0, timestamp=t)


def test_track_range_calculation():
    """Test that track range is calculated correctly."""
    kf = KalmanFilter()
    state, cov = kf.initialize_state(3.0, 4.0, 0.0)
    
    track = Track(
        track_id=1,
        x=3.0, y=4.0, z=0.0,
        vx=0.0, vy=0.0, vz=0.0,
        snr=15.0,
        classification=ObjectClass.UNKNOWN,
        confidence=0.5,
        hits=5,
        misses=0,
        history=[],
        last_update=0.0,
        state=state,
        covariance=cov
    )
    
    # 3-4-5 triangle
    assert np.isclose(track.range, 5.0, atol=0.001), f"Range should be 5.0, got {track.range}"
    print("✓ Track range calculation test passed")


def test_track_azimuth_calculation():
    """Test that track azimuth is calculated correctly."""
    kf = KalmanFilter()
    
    test_cases = [
        # (x, y, expected_azimuth_deg)
        (0.0, 2.0, 0.0),    # Straight ahead
        (2.0, 2.0, 45.0),   # 45° right
        (-2.0, 2.0, -45.0), # 45° left
        (1.0, 1.732, 30.0), # 30° right (tan(30°) ≈ 0.577)
    ]
    
    for x, y, expected_az in test_cases:
        state, cov = kf.initialize_state(x, y, 0.0)
        track = Track(
            track_id=1, x=x, y=y, z=0.0,
            vx=0.0, vy=0.0, vz=0.0,
            snr=15.0, classification=ObjectClass.UNKNOWN,
            confidence=0.5, hits=5, misses=0,
            history=[], last_update=0.0,
            state=state, covariance=cov
        )
        
        assert np.isclose(track.azimuth_deg, expected_az, atol=1.0), \
            f"Azimuth for ({x}, {y}) should be {expected_az}°, got {track.azimuth_deg}°"
    
    print("✓ Track azimuth calculation test passed")


def test_track_elevation_calculation():
    """Test that track elevation is calculated correctly."""
    kf = KalmanFilter()
    
    # Object at y=2, z=2 should be at 45° elevation
    state, cov = kf.initialize_state(0.0, 2.0, 2.0)
    track = Track(
        track_id=1, x=0.0, y=2.0, z=2.0,
        vx=0.0, vy=0.0, vz=0.0,
        snr=15.0, classification=ObjectClass.UNKNOWN,
        confidence=0.5, hits=5, misses=0,
        history=[], last_update=0.0,
        state=state, covariance=cov
    )
    
    assert np.isclose(track.elevation_deg, 45.0, atol=1.0), \
        f"Elevation should be 45°, got {track.elevation_deg}°"
    
    print("✓ Track elevation calculation test passed")


def test_track_velocity_magnitude():
    """Test that velocity magnitude is calculated correctly."""
    kf = KalmanFilter()
    state, cov = kf.initialize_state(0.0, 0.0, 0.0)
    
    # Set velocity to (3, 4, 0) - magnitude should be 5
    state[3] = 3.0
    state[4] = 4.0
    state[5] = 0.0
    
    track = Track(
        track_id=1, x=0.0, y=0.0, z=0.0,
        vx=3.0, vy=4.0, vz=0.0,
        snr=15.0, classification=ObjectClass.UNKNOWN,
        confidence=0.5, hits=5, misses=0,
        history=[], last_update=0.0,
        state=state, covariance=cov
    )
    
    assert np.isclose(track.velocity_magnitude, 5.0, atol=0.001), \
        f"Velocity magnitude should be 5.0, got {track.velocity_magnitude}"
    
    print("✓ Track velocity magnitude test passed")


def test_track_distance_to():
    """Test that distance_to calculation works correctly."""
    kf = KalmanFilter()
    state, cov = kf.initialize_state(1.0, 2.0, 3.0)
    
    track = Track(
        track_id=1, x=1.0, y=2.0, z=3.0,
        vx=0.0, vy=0.0, vz=0.0,
        snr=15.0, classification=ObjectClass.UNKNOWN,
        confidence=0.5, hits=5, misses=0,
        history=[], last_update=0.0,
        state=state, covariance=cov
    )
    
    # Distance to itself should be 0
    assert np.isclose(track.distance_to(1.0, 2.0, 3.0), 0.0, atol=0.001)
    
    # Distance to (4, 6, 3) should be 5 (3-4-5 triangle in XY plane)
    assert np.isclose(track.distance_to(4.0, 6.0, 3.0), 5.0, atol=0.001)
    
    print("✓ Track distance_to test passed")


def test_confidence_lifecycle():
    """Test that confidence increases and decreases correctly."""
    config = PipelineConfig()
    
    initial = config.initial_confidence
    increment = config.confidence_increment
    decay = config.confidence_decay
    
    # Starting confidence
    confidence = initial
    assert confidence == 0.3, f"Initial confidence should be 0.3, got {confidence}"
    
    # After 5 hits
    for _ in range(5):
        confidence = min(1.0, confidence + increment)
    
    expected = min(1.0, initial + 5 * increment)
    assert np.isclose(confidence, expected, atol=0.01), \
        f"After 5 hits confidence should be {expected}, got {confidence}"
    
    # After 10 misses (decay)
    for _ in range(10):
        confidence = max(0.0, confidence - decay)
    
    expected = max(0.0, expected - 10 * decay)
    assert np.isclose(confidence, expected, atol=0.01), \
        f"After 10 misses confidence should be {expected}, got {confidence}"
    
    print("✓ Confidence lifecycle test passed")


def test_world_state_creation():
    """Test that WorldState is created correctly."""
    from radar_pipeline.data_types import RadarHealth, StaticObject
    
    kf = KalmanFilter()
    state, cov = kf.initialize_state(1.0, 2.0, 0.5)
    
    tracks = [
        Track(
            track_id=1, x=1.0, y=2.0, z=0.5,
            vx=0.1, vy=0.2, vz=0.0,
            snr=15.0, classification=ObjectClass.DYNAMIC,
            confidence=0.8, hits=10, misses=0,
            history=[(1.0, 2.0, 0.5, 0.0)], last_update=0.0,
            state=state, covariance=cov
        )
    ]
    
    static_objects = [
        StaticObject(x=5.0, y=3.0, z=0.0, radius=0.1, 
                     detections=50, first_seen=0.0, last_seen=1.0)
    ]
    
    health = RadarHealth(fps=20.0, active_tracks=1, static_objects=1)
    
    world_state = WorldState(
        timestamp=1.0,
        frame_number=100,
        tracks=tracks,
        static_objects=static_objects,
        occupancy_grid=None,
        radar_health=health
    )
    
    assert world_state.frame_number == 100
    assert len(world_state.tracks) == 1
    assert len(world_state.static_objects) == 1
    
    # Test helper methods
    closest = world_state.get_closest_track()
    assert closest is not None
    assert closest.track_id == 1
    
    dynamic_tracks = world_state.get_tracks_by_classification(ObjectClass.DYNAMIC)
    assert len(dynamic_tracks) == 1
    
    print("✓ WorldState creation test passed")


def run_all_tests():
    """Run all tracking tests."""
    print("\n" + "="*50)
    print("TRACKING TESTS")
    print("="*50 + "\n")
    
    test_track_range_calculation()
    test_track_azimuth_calculation()
    test_track_elevation_calculation()
    test_track_velocity_magnitude()
    test_track_distance_to()
    test_confidence_lifecycle()
    test_world_state_creation()
    
    print("\n" + "="*50)
    print("ALL TRACKING TESTS PASSED ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()

