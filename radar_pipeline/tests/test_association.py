"""
Tests for track-detection association logic.
"""

import numpy as np
import sys
sys.path.insert(0, '../..')

from radar_pipeline.data_types import Detection, Track, ObjectClass
from radar_pipeline.kalman import KalmanFilter
from radar_pipeline.association import (
    associate, 
    greedy_association, 
    compute_cost_matrix,
    simple_euclidean_association
)


def create_track(track_id: int, x: float, y: float, z: float, kf: KalmanFilter) -> Track:
    """Helper to create a track at a given position."""
    state, cov = kf.initialize_state(x, y, z)
    return Track(
        track_id=track_id,
        x=x, y=y, z=z,
        vx=0.0, vy=0.0, vz=0.0,
        snr=15.0,
        classification=ObjectClass.UNKNOWN,
        confidence=0.5,
        hits=5,
        misses=0,
        history=[(x, y, z, 0.0)],
        last_update=0.0,
        state=state,
        covariance=cov
    )


def create_detection(x: float, y: float, z: float) -> Detection:
    """Helper to create a detection at a given position."""
    return Detection(x=x, y=y, z=z, snr=15.0, timestamp=0.05)


def test_association_perfect_match():
    """Test that nearby detections associate with tracks."""
    kf = KalmanFilter()
    
    # Create tracks at known positions
    tracks = [
        create_track(1, 1.0, 2.0, 0.0, kf),
        create_track(2, 3.0, 4.0, 0.5, kf),
    ]
    
    # Create detections near those positions
    detections = [
        create_detection(1.05, 2.02, 0.0),  # Near track 1
        create_detection(3.08, 4.05, 0.48),  # Near track 2
    ]
    
    matches, unmatched_t, unmatched_d = associate(tracks, detections, kf, gate_threshold=0.5)
    
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
    assert len(unmatched_t) == 0, f"Expected 0 unmatched tracks, got {len(unmatched_t)}"
    assert len(unmatched_d) == 0, f"Expected 0 unmatched detections, got {len(unmatched_d)}"
    
    print("✓ Perfect match association test passed")


def test_association_new_detection():
    """Test that new detections (far from tracks) are unmatched."""
    kf = KalmanFilter()
    
    tracks = [create_track(1, 1.0, 2.0, 0.0, kf)]
    
    detections = [
        create_detection(1.05, 2.02, 0.0),  # Near track 1
        create_detection(5.0, 6.0, 0.0),     # Far away - new detection
    ]
    
    matches, unmatched_t, unmatched_d = associate(tracks, detections, kf, gate_threshold=0.5)
    
    assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"
    assert len(unmatched_d) == 1, f"Expected 1 unmatched detection, got {len(unmatched_d)}"
    assert 1 in unmatched_d, "Detection index 1 should be unmatched"
    
    print("✓ New detection association test passed")


def test_association_missed_track():
    """Test that tracks without detections are unmatched."""
    kf = KalmanFilter()
    
    tracks = [
        create_track(1, 1.0, 2.0, 0.0, kf),
        create_track(2, 3.0, 4.0, 0.0, kf),  # This one won't have a detection
    ]
    
    detections = [
        create_detection(1.05, 2.02, 0.0),  # Only near track 1
    ]
    
    matches, unmatched_t, unmatched_d = associate(tracks, detections, kf, gate_threshold=0.5)
    
    assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"
    assert len(unmatched_t) == 1, f"Expected 1 unmatched track, got {len(unmatched_t)}"
    assert 1 in unmatched_t, "Track index 1 should be unmatched"
    
    print("✓ Missed track association test passed")


def test_association_gate_threshold():
    """Test that gate threshold correctly filters associations."""
    kf = KalmanFilter()
    
    tracks = [create_track(1, 1.0, 2.0, 0.0, kf)]
    
    # Detection 0.6m away - Mahalanobis distance will be ~1.8 due to covariance
    detections = [create_detection(1.6, 2.0, 0.0)]
    
    # With 1.0 Mahalanobis gate, should NOT match (Mahalanobis dist is ~1.8)
    matches, _, _ = associate(tracks, detections, kf, gate_threshold=1.0)
    assert len(matches) == 0, "Should not match with 1.0 Mahalanobis gate"
    
    # With 2.0 Mahalanobis gate, SHOULD match
    matches, _, _ = associate(tracks, detections, kf, gate_threshold=2.0)
    assert len(matches) == 1, "Should match with 2.0 Mahalanobis gate"
    
    print("✓ Gate threshold test passed")


def test_association_z_dimension():
    """Test that Z dimension is properly considered in association."""
    kf = KalmanFilter()
    
    # Track at Z=0
    tracks = [create_track(1, 1.0, 2.0, 0.0, kf)]
    
    # Detection at same X,Y but different Z (0.8m higher)
    # Mahalanobis distance will be ~2.4 due to covariance
    detections = [create_detection(1.0, 2.0, 0.8)]
    
    # Should NOT match with 1.5 Mahalanobis gate
    matches, _, _ = associate(tracks, detections, kf, gate_threshold=1.5)
    assert len(matches) == 0, "Z difference should prevent match with tight gate"
    
    # Should match with 3.0 Mahalanobis gate
    matches, _, _ = associate(tracks, detections, kf, gate_threshold=3.0)
    assert len(matches) == 1, "Should match with larger gate"
    
    print("✓ Z dimension association test passed")


def test_association_no_tracks():
    """Test association when there are no existing tracks."""
    kf = KalmanFilter()
    
    tracks = []
    detections = [
        create_detection(1.0, 2.0, 0.0),
        create_detection(3.0, 4.0, 0.5),
    ]
    
    matches, unmatched_t, unmatched_d = associate(tracks, detections, kf, gate_threshold=0.5)
    
    assert len(matches) == 0, "No matches possible with no tracks"
    assert len(unmatched_d) == 2, "All detections should be unmatched"
    
    print("✓ No tracks association test passed")


def test_association_no_detections():
    """Test association when there are no detections."""
    kf = KalmanFilter()
    
    tracks = [
        create_track(1, 1.0, 2.0, 0.0, kf),
        create_track(2, 3.0, 4.0, 0.5, kf),
    ]
    detections = []
    
    matches, unmatched_t, unmatched_d = associate(tracks, detections, kf, gate_threshold=0.5)
    
    assert len(matches) == 0, "No matches possible with no detections"
    assert len(unmatched_t) == 2, "All tracks should be unmatched"
    
    print("✓ No detections association test passed")


def run_all_tests():
    """Run all association tests."""
    print("\n" + "="*50)
    print("ASSOCIATION TESTS")
    print("="*50 + "\n")
    
    test_association_perfect_match()
    test_association_new_detection()
    test_association_missed_track()
    test_association_gate_threshold()
    test_association_z_dimension()
    test_association_no_tracks()
    test_association_no_detections()
    
    print("\n" + "="*50)
    print("ALL ASSOCIATION TESTS PASSED ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()

