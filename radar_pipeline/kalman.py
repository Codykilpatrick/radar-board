"""
Kalman filter implementation for radar track state estimation.

State vector: [x, y, z, vx, vy, vz]
Constant velocity motion model with position-only measurements.
"""

import numpy as np
from typing import Tuple


class KalmanFilter:
    """
    6-state Kalman filter for 3D position and velocity tracking.
    
    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """
    
    def __init__(self, 
                 dt: float = 0.05,
                 process_noise: float = 0.1,
                 measurement_noise: float = 0.01):
        """
        Initialize Kalman filter matrices.
        
        Args:
            dt: Time step (frame period)
            process_noise: Process noise scaling factor
            measurement_noise: Measurement noise scaling factor
        """
        self.dt = dt
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=np.float64)
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)
        
        # Process noise covariance
        # Accounts for acceleration uncertainty
        self.Q = self._create_process_noise(dt, process_noise)
        
        # Measurement noise covariance
        self.R = np.eye(3, dtype=np.float64) * measurement_noise
    
    def _create_process_noise(self, dt: float, q: float) -> np.ndarray:
        """
        Create process noise matrix using discrete white noise acceleration model.
        
        This models random accelerations affecting the velocity.
        """
        # Simplified diagonal process noise
        # Position uncertainty grows with velocity uncertainty
        Q = np.array([
            [dt**4/4, 0,       0,       dt**3/2, 0,       0],
            [0,       dt**4/4, 0,       0,       dt**3/2, 0],
            [0,       0,       dt**4/4, 0,       0,       dt**3/2],
            [dt**3/2, 0,       0,       dt**2,   0,       0],
            [0,       dt**3/2, 0,       0,       dt**2,   0],
            [0,       0,       dt**3/2, 0,       0,       dt**2],
        ], dtype=np.float64) * q
        
        return Q
    
    def initialize_state(self, x: float, y: float, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state and covariance for a new track.
        
        Args:
            x, y, z: Initial position
        
        Returns:
            Tuple of (state, covariance)
        """
        state = np.array([x, y, z, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Initial covariance - high uncertainty in velocity
        covariance = np.diag([
            0.1,   # x position uncertainty
            0.1,   # y position uncertainty
            0.1,   # z position uncertainty
            1.0,   # vx velocity uncertainty (high, unknown)
            1.0,   # vy velocity uncertainty
            1.0,   # vz velocity uncertainty
        ]).astype(np.float64)
        
        return state, covariance
    
    def predict(self, state: np.ndarray, covariance: np.ndarray, 
                dt: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman prediction step.
        
        Args:
            state: Current state vector (6,)
            covariance: Current covariance matrix (6, 6)
            dt: Time step override (uses default if None)
        
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        if dt is not None and dt != self.dt:
            # Update F matrix for different dt
            F = self.F.copy()
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
            Q = self._create_process_noise(dt, self.Q[3, 3] / (self.dt**2))
        else:
            F = self.F
            Q = self.Q
        
        # Predict state
        predicted_state = F @ state
        
        # Predict covariance
        predicted_covariance = F @ covariance @ F.T + Q
        
        return predicted_state, predicted_covariance
    
    def update(self, state: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step.
        
        Args:
            state: Predicted state vector (6,)
            covariance: Predicted covariance matrix (6, 6)
            measurement: Measurement vector [x, y, z] (3,)
        
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ state
        
        # Innovation covariance
        S = self.H @ covariance @ self.H.T + self.R
        
        # Kalman gain
        try:
            K = covariance @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            K = covariance @ self.H.T @ np.linalg.pinv(S)
        
        # Update state
        updated_state = state + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I = np.eye(6)
        IKH = I - K @ self.H
        updated_covariance = IKH @ covariance @ IKH.T + K @ self.R @ K.T
        
        return updated_state, updated_covariance
    
    def get_innovation_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """
        Compute innovation covariance for gating.
        
        Args:
            covariance: State covariance matrix (6, 6)
        
        Returns:
            Innovation covariance matrix (3, 3)
        """
        return self.H @ covariance @ self.H.T + self.R


# Global filter instance for default parameters
_default_filter = None


def get_default_filter(dt: float = 0.05,
                       process_noise: float = 0.1,
                       measurement_noise: float = 0.01) -> KalmanFilter:
    """Get or create the default Kalman filter instance."""
    global _default_filter
    if _default_filter is None:
        _default_filter = KalmanFilter(dt, process_noise, measurement_noise)
    return _default_filter

