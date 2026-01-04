"""
Track-detection association logic.

Implements nearest-neighbor association with gating using Mahalanobis distance.
"""

import numpy as np
from typing import List, Tuple, Set
from .data_types import Detection, Track
from .kalman import KalmanFilter


def compute_cost_matrix(tracks: List[Track],
                        detections: List[Detection],
                        kalman: KalmanFilter) -> np.ndarray:
    """
    Compute cost matrix for track-detection association.
    
    Uses Mahalanobis distance between predicted track positions and detections.
    
    Args:
        tracks: List of existing tracks
        detections: List of new detections
        kalman: Kalman filter instance for covariance access
    
    Returns:
        Cost matrix of shape (num_tracks, num_detections)
    """
    if not tracks or not detections:
        return np.array([])
    
    num_tracks = len(tracks)
    num_detections = len(detections)
    
    cost_matrix = np.full((num_tracks, num_detections), np.inf)
    
    for i, track in enumerate(tracks):
        # Get innovation covariance for this track
        S = kalman.get_innovation_covariance(track.covariance)
        
        try:
            S_inv = np.linalg.inv(S + np.eye(3) * 1e-6)
        except np.linalg.LinAlgError:
            S_inv = np.eye(3)
        
        for j, det in enumerate(detections):
            # Innovation vector
            innovation = np.array([
                det.x - track.state[0],
                det.y - track.state[1],
                det.z - track.state[2]
            ])
            
            # Mahalanobis distance
            distance = np.sqrt(innovation.T @ S_inv @ innovation)
            cost_matrix[i, j] = distance
    
    return cost_matrix


def greedy_association(cost_matrix: np.ndarray,
                       gate_threshold: float) -> List[Tuple[int, int]]:
    """
    Greedy nearest-neighbor association with gating.
    
    Args:
        cost_matrix: Cost matrix of shape (num_tracks, num_detections)
        gate_threshold: Maximum distance for valid association
    
    Returns:
        List of (track_idx, detection_idx) pairs
    """
    if cost_matrix.size == 0:
        return []
    
    num_tracks, num_detections = cost_matrix.shape
    
    # Track which indices are still available
    available_tracks = set(range(num_tracks))
    available_detections = set(range(num_detections))
    
    matches = []
    
    # Get all valid (under threshold) costs in sorted order
    valid_costs = []
    for i in range(num_tracks):
        for j in range(num_detections):
            if cost_matrix[i, j] < gate_threshold:
                valid_costs.append((cost_matrix[i, j], i, j))
    
    valid_costs.sort(key=lambda x: x[0])
    
    # Greedily assign lowest costs first
    for cost, track_idx, det_idx in valid_costs:
        if track_idx in available_tracks and det_idx in available_detections:
            matches.append((track_idx, det_idx))
            available_tracks.remove(track_idx)
            available_detections.remove(det_idx)
    
    return matches


def hungarian_association(cost_matrix: np.ndarray,
                          gate_threshold: float) -> List[Tuple[int, int]]:
    """
    Hungarian algorithm for optimal track-detection association.
    
    Falls back to greedy if scipy is not available.
    
    Args:
        cost_matrix: Cost matrix of shape (num_tracks, num_detections)
        gate_threshold: Maximum distance for valid association
    
    Returns:
        List of (track_idx, detection_idx) pairs
    """
    if cost_matrix.size == 0:
        return []
    
    try:
        from scipy.optimize import linear_sum_assignment
        
        # Replace inf with large value for Hungarian algorithm
        cost_work = cost_matrix.copy()
        cost_work[cost_work > gate_threshold] = 1e6
        
        row_indices, col_indices = linear_sum_assignment(cost_work)
        
        # Filter out assignments that exceed the gate
        matches = []
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < gate_threshold:
                matches.append((r, c))
        
        return matches
        
    except ImportError:
        # Fall back to greedy if scipy not available
        return greedy_association(cost_matrix, gate_threshold)


def associate(tracks: List[Track],
              detections: List[Detection],
              kalman: KalmanFilter,
              gate_threshold: float = 0.5,
              use_hungarian: bool = True) -> Tuple[List[Tuple[int, int]], 
                                                    Set[int], 
                                                    Set[int]]:
    """
    Associate detections to existing tracks.
    
    Args:
        tracks: List of existing tracks
        detections: List of new detections
        kalman: Kalman filter instance
        gate_threshold: Maximum Mahalanobis distance for association
        use_hungarian: Use Hungarian algorithm (optimal) vs greedy
    
    Returns:
        Tuple of:
            - List of (track_idx, detection_idx) matched pairs
            - Set of unmatched track indices
            - Set of unmatched detection indices
    """
    if not tracks:
        return [], set(), set(range(len(detections)))
    
    if not detections:
        return [], set(range(len(tracks))), set()
    
    # Compute cost matrix
    cost_matrix = compute_cost_matrix(tracks, detections, kalman)
    
    # Find associations
    if use_hungarian:
        matches = hungarian_association(cost_matrix, gate_threshold)
    else:
        matches = greedy_association(cost_matrix, gate_threshold)
    
    # Find unmatched tracks and detections
    matched_tracks = {m[0] for m in matches}
    matched_detections = {m[1] for m in matches}
    
    unmatched_tracks = set(range(len(tracks))) - matched_tracks
    unmatched_detections = set(range(len(detections))) - matched_detections
    
    return matches, unmatched_tracks, unmatched_detections


def simple_euclidean_association(tracks: List[Track],
                                  detections: List[Detection],
                                  gate_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], 
                                                                         Set[int], 
                                                                         Set[int]]:
    """
    Simple Euclidean distance association (no Kalman covariance).
    
    Useful for initial testing or when Kalman filter is not yet converged.
    
    Args:
        tracks: List of existing tracks
        detections: List of new detections
        gate_threshold: Maximum Euclidean distance for association (meters)
    
    Returns:
        Tuple of:
            - List of (track_idx, detection_idx) matched pairs
            - Set of unmatched track indices
            - Set of unmatched detection indices
    """
    if not tracks:
        return [], set(), set(range(len(detections)))
    
    if not detections:
        return [], set(range(len(tracks))), set()
    
    num_tracks = len(tracks)
    num_detections = len(detections)
    
    # Compute Euclidean distance matrix
    cost_matrix = np.zeros((num_tracks, num_detections))
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = track.distance_to(det.x, det.y, det.z)
    
    # Use greedy association
    matches = greedy_association(cost_matrix, gate_threshold)
    
    matched_tracks = {m[0] for m in matches}
    matched_detections = {m[1] for m in matches}
    
    unmatched_tracks = set(range(len(tracks))) - matched_tracks
    unmatched_detections = set(range(len(detections))) - matched_detections
    
    return matches, unmatched_tracks, unmatched_detections

