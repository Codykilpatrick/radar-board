"""
Tests for Kalman filter implementation.
"""

import numpy as np
import sys
sys.path.insert(0, '../..')

from radar_pipeline.kalman import KalmanFilter


def test_kalman_initialization():
    """Test that Kalman filter initializes state correctly."""
    kf = KalmanFilter(dt=0.05)
    
    state, cov = kf.initialize_state(1.0, 2.0, 0.5)
    
    assert state.shape == (6,), f"State shape wrong: {state.shape}"
    assert cov.shape == (6, 6), f"Covariance shape wrong: {cov.shape}"
    assert np.allclose(state[:3], [1.0, 2.0, 0.5]), "Position not initialized correctly"
    assert np.allclose(state[3:], [0.0, 0.0, 0.0]), "Velocity should be zero initially"
    print("✓ Kalman initialization test passed")


def test_kalman_prediction():
    """Test that prediction moves state forward correctly."""
    kf = KalmanFilter(dt=0.05)
    
    # Start at (1, 2, 0.5) with velocity (1, 0, 0) - moving in X
    state = np.array([1.0, 2.0, 0.5, 1.0, 0.0, 0.0])
    cov = np.eye(6) * 0.1
    
    # Predict forward
    new_state, new_cov = kf.predict(state, cov)
    
    # Position should move by velocity * dt
    expected_x = 1.0 + 1.0 * 0.05  # 1.05
    assert np.isclose(new_state[0], expected_x, atol=0.001), f"X prediction wrong: {new_state[0]} != {expected_x}"
    assert np.isclose(new_state[1], 2.0, atol=0.001), "Y should not change"
    assert np.isclose(new_state[2], 0.5, atol=0.001), "Z should not change"
    
    # Velocity should remain same (constant velocity model)
    assert np.isclose(new_state[3], 1.0, atol=0.001), "Vx should remain 1.0"
    
    # Covariance should grow
    assert np.all(np.diag(new_cov) >= np.diag(cov) - 0.001), "Covariance should not shrink on predict"
    
    print("✓ Kalman prediction test passed")


def test_kalman_update():
    """Test that update corrects state toward measurement."""
    kf = KalmanFilter(dt=0.05, measurement_noise=0.01)
    
    # State thinks we're at (1, 2, 0.5)
    state = np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.0])
    cov = np.eye(6) * 0.1
    
    # Measurement says we're at (1.1, 2.0, 0.5)
    measurement = np.array([1.1, 2.0, 0.5])
    
    new_state, new_cov = kf.update(state, cov, measurement)
    
    # State should move toward measurement
    assert new_state[0] > 1.0, "X should increase toward measurement"
    assert new_state[0] < 1.1, "X should not overshoot measurement"
    
    # Covariance should shrink (we got new information)
    assert new_cov[0, 0] < cov[0, 0], "Covariance should decrease after update"
    
    print("✓ Kalman update test passed")


def test_kalman_velocity_estimation():
    """Test that Kalman filter estimates velocity from position updates."""
    kf = KalmanFilter(dt=0.05, process_noise=0.5, measurement_noise=0.01)
    
    state, cov = kf.initialize_state(0.0, 0.0, 0.0)
    
    # Simulate object moving at 1 m/s in Y direction
    velocity_y = 1.0  # m/s
    
    for i in range(20):
        t = i * 0.05
        true_y = velocity_y * t
        
        # Predict
        state, cov = kf.predict(state, cov)
        
        # Measurement (with small noise)
        measurement = np.array([0.0, true_y + np.random.randn() * 0.02, 0.0])
        
        # Update
        state, cov = kf.update(state, cov, measurement)
    
    # After 20 frames, velocity estimate should be close to 1.0 m/s
    estimated_vy = state[4]
    assert abs(estimated_vy - velocity_y) < 0.3, f"Velocity estimate {estimated_vy} too far from {velocity_y}"
    
    print(f"✓ Kalman velocity estimation test passed (estimated vy={estimated_vy:.2f}, true=1.0)")


def test_kalman_coasting():
    """Test that Kalman filter can coast (predict without updates)."""
    kf = KalmanFilter(dt=0.05)
    
    # Object at (1, 2, 0.5) moving at (0.5, 1.0, 0) m/s
    state = np.array([1.0, 2.0, 0.5, 0.5, 1.0, 0.0])
    initial_cov = np.eye(6) * 0.01  # Low uncertainty - we're confident
    cov = initial_cov.copy()
    
    # Coast for 10 frames (0.5 seconds) without any measurements
    for _ in range(10):
        state, cov = kf.predict(state, cov)
    
    # Position should have moved by velocity * time
    expected_x = 1.0 + 0.5 * 0.5  # 1.25
    expected_y = 2.0 + 1.0 * 0.5  # 2.5
    
    assert np.isclose(state[0], expected_x, atol=0.01), f"Coasted X wrong: {state[0]} != {expected_x}"
    assert np.isclose(state[1], expected_y, atol=0.01), f"Coasted Y wrong: {state[1]} != {expected_y}"
    
    # Uncertainty should have grown (at least 20% increase)
    assert cov[0, 0] > initial_cov[0, 0] * 1.2, \
        f"Covariance should grow during coasting: {initial_cov[0,0]} -> {cov[0,0]}"
    
    print(f"✓ Kalman coasting test passed (coasted to x={state[0]:.2f}, y={state[1]:.2f}, cov grew {cov[0,0]/initial_cov[0,0]:.1f}x)")


def run_all_tests():
    """Run all Kalman filter tests."""
    print("\n" + "="*50)
    print("KALMAN FILTER TESTS")
    print("="*50 + "\n")
    
    test_kalman_initialization()
    test_kalman_prediction()
    test_kalman_update()
    test_kalman_velocity_estimation()
    test_kalman_coasting()
    
    print("\n" + "="*50)
    print("ALL KALMAN TESTS PASSED ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()

