"""
Utility functions for the radar pipeline.
"""

import numpy as np
from typing import Tuple


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical (range, azimuth, elevation).
    
    Args:
        x: X coordinate (left/right)
        y: Y coordinate (forward)
        z: Z coordinate (up/down)
    
    Returns:
        Tuple of (range_m, azimuth_deg, elevation_deg)
    """
    range_m = np.sqrt(x**2 + y**2 + z**2)
    azimuth_rad = np.arctan2(x, y)
    xy_dist = np.sqrt(x**2 + y**2)
    elevation_rad = np.arctan2(z, xy_dist)
    
    return range_m, np.rad2deg(azimuth_rad), np.rad2deg(elevation_rad)


def spherical_to_cartesian(range_m: float, azimuth_deg: float, elevation_deg: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian.
    
    Args:
        range_m: Distance in meters
        azimuth_deg: Azimuth angle in degrees
        elevation_deg: Elevation angle in degrees
    
    Returns:
        Tuple of (x, y, z)
    """
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    
    xy_dist = range_m * np.cos(elevation_rad)
    x = xy_dist * np.sin(azimuth_rad)
    y = xy_dist * np.cos(azimuth_rad)
    z = range_m * np.sin(elevation_rad)
    
    return x, y, z


def is_in_fov(x: float, y: float, z: float, 
              min_range: float, max_range: float, 
              fov_angle: float) -> bool:
    """
    Check if a point is within the radar's field of view.
    
    Args:
        x, y, z: Cartesian coordinates
        min_range: Minimum detection range
        max_range: Maximum detection range
        fov_angle: Half-angle of FOV in degrees
    
    Returns:
        True if point is within FOV
    """
    range_m = np.sqrt(x**2 + y**2 + z**2)
    if range_m < min_range or range_m > max_range:
        return False
    
    azimuth_deg = np.rad2deg(np.arctan2(x, y)) if y > 0 else 90.0
    if abs(azimuth_deg) > fov_angle:
        return False
    
    return True


def compute_distance(x1: float, y1: float, z1: float,
                    x2: float, y2: float, z2: float) -> float:
    """Compute 3D Euclidean distance between two points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def mahalanobis_distance(detection: np.ndarray, 
                         predicted_state: np.ndarray,
                         covariance: np.ndarray,
                         H: np.ndarray) -> float:
    """
    Compute Mahalanobis distance for track-detection association.
    
    Args:
        detection: Measurement vector [x, y, z]
        predicted_state: Predicted state vector [x, y, z, vx, vy, vz]
        covariance: State covariance matrix (6x6)
        H: Measurement matrix (3x6)
    
    Returns:
        Mahalanobis distance
    """
    # Innovation (measurement residual)
    predicted_measurement = H @ predicted_state
    innovation = detection - predicted_measurement
    
    # Innovation covariance
    S = H @ covariance @ H.T
    
    # Add small regularization for numerical stability
    S += np.eye(3) * 1e-6
    
    try:
        S_inv = np.linalg.inv(S)
        distance = np.sqrt(innovation.T @ S_inv @ innovation)
    except np.linalg.LinAlgError:
        # Fallback to Euclidean distance if matrix is singular
        distance = np.linalg.norm(innovation)
    
    return distance


def color_by_range(range_m: float) -> Tuple[float, float, float, float]:
    """
    Get RGBA color based on range.
    
    Red: < 1m
    Yellow: 1-3m
    Green: > 3m
    """
    if range_m < 1.0:
        return (1.0, 0.3, 0.3, 1.0)  # Red
    elif range_m < 3.0:
        return (1.0, 1.0, 0.3, 1.0)  # Yellow
    else:
        return (0.3, 1.0, 0.3, 1.0)  # Green


def color_by_classification(classification) -> Tuple[float, float, float, float]:
    """
    Get RGBA color based on object classification.
    """
    from .data_types import ObjectClass
    
    if classification == ObjectClass.STATIC:
        return (0.5, 0.5, 0.5, 0.8)  # Gray
    elif classification == ObjectClass.DYNAMIC:
        return (0.3, 0.8, 1.0, 1.0)  # Cyan
    else:
        return (1.0, 0.5, 0.0, 1.0)  # Orange


def create_occupancy_grid(grid_size: Tuple[int, int], 
                          resolution: float) -> np.ndarray:
    """
    Create an empty occupancy grid.
    
    Args:
        grid_size: (width, height) in cells
        resolution: meters per cell
    
    Returns:
        2D numpy array of zeros
    """
    return np.zeros(grid_size, dtype=np.float32)


def update_occupancy_grid(grid: np.ndarray,
                          x: float, y: float,
                          resolution: float,
                          decay: float = 0.95) -> None:
    """
    Update occupancy grid with a detection.
    
    Args:
        grid: Occupancy grid to update (modified in place)
        x, y: Detection coordinates
        resolution: Meters per cell
        decay: Decay factor for existing values
    """
    # Apply decay to entire grid
    grid *= decay
    
    # Convert to grid coordinates (center of grid is origin)
    grid_x = int(x / resolution + grid.shape[0] / 2)
    grid_y = int(y / resolution + grid.shape[1] / 2)
    
    # Check bounds
    if 0 <= grid_x < grid.shape[0] and 0 <= grid_y < grid.shape[1]:
        grid[grid_x, grid_y] = min(1.0, grid[grid_x, grid_y] + 0.2)

