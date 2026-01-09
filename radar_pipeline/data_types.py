"""
Data structures for the radar pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np


class ObjectClass(Enum):
    """Classification of tracked objects."""
    UNKNOWN = 0
    DYNAMIC = 1
    STATIC = 2


@dataclass
class Detection:
    """Single radar detection from one frame."""
    x: float
    y: float
    z: float
    snr: float
    timestamp: float


@dataclass
class Track:
    """Tracked object with state estimation."""
    track_id: int
    x: float
    y: float
    z: float
    vx: float  # velocity estimates
    vy: float
    vz: float
    snr: float
    classification: ObjectClass
    confidence: float  # 0-1, increases with confirmations
    hits: int  # consecutive detection count
    misses: int  # consecutive missed frames
    history: List[Tuple[float, float, float, float]]  # [(x, y, z, timestamp), ...]
    last_update: float
    
    # Kalman filter state (6-state: x, y, z, vx, vy, vz)
    state: np.ndarray  # shape (6,)
    covariance: np.ndarray  # shape (6, 6)
    
    # Track for static classification
    low_velocity_frames: int = 0
    
    # Once confirmed, track stays visible until deleted
    confirmed: bool = False
    
    @property
    def range(self) -> float:
        """Distance from radar to track."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    @property
    def azimuth_deg(self) -> float:
        """Azimuth angle in degrees."""
        return np.rad2deg(np.arctan2(self.x, self.y)) if self.y > 0 else 90.0
    
    @property
    def elevation_deg(self) -> float:
        """Elevation angle in degrees."""
        xy_dist = np.sqrt(self.x**2 + self.y**2)
        return np.rad2deg(np.arctan2(self.z, xy_dist))
    
    @property
    def velocity_magnitude(self) -> float:
        """Speed of the tracked object."""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def distance_to(self, x: float, y: float, z: float) -> float:
        """3D Euclidean distance to a point."""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)

    # ===== Threat Assessment Properties =====

    @property
    def closing_velocity(self) -> float:
        """
        Rate of range change (m/s). Positive = approaching, negative = receding.

        Computed as the radial component of velocity (dot product of velocity
        with unit vector pointing from radar to target).
        """
        r = self.range
        if r < 0.001:
            return 0.0
        # Radial velocity = -(v · r_hat) where r_hat points from radar to target
        # Negative sign because we want positive = approaching
        return -(self.x * self.vx + self.y * self.vy + self.z * self.vz) / r

    @property
    def time_to_intercept(self) -> float:
        """
        Estimated time until target reaches radar origin (seconds).

        Returns float('inf') if target is not approaching.
        Returns 0 if target is already at origin.
        """
        cv = self.closing_velocity
        if cv <= 0:
            return float('inf')  # Not approaching
        r = self.range
        if r < 0.01:
            return 0.0
        return r / cv

    @property
    def threat_score(self) -> float:
        """
        Threat score from 0-100. Higher = more threatening.

        Factors:
        - Closing velocity (approaching = higher threat)
        - Time to intercept (sooner = higher threat)
        - Range (closer = higher threat)
        - Track confidence (higher confidence = more reliable threat)
        """
        score = 0.0

        # Closing velocity component (0-40 points)
        # Max score at 5+ m/s closing velocity
        cv = self.closing_velocity
        if cv > 0:
            score += min(40, cv * 8)

        # Time to intercept component (0-30 points)
        # Max score if intercept < 1 second
        tti = self.time_to_intercept
        if tti < float('inf'):
            if tti < 1.0:
                score += 30
            elif tti < 5.0:
                score += 30 * (1 - (tti - 1.0) / 4.0)

        # Range component (0-20 points)
        # Max score if < 1m
        r = self.range
        if r < 1.0:
            score += 20
        elif r < 5.0:
            score += 20 * (1 - (r - 1.0) / 4.0)

        # Confidence component (0-10 points)
        score += self.confidence * 10

        return min(100, score)

    @property
    def is_approaching(self) -> bool:
        """True if target is approaching the radar."""
        return self.closing_velocity > 0.1  # Small threshold to avoid noise

    @property
    def is_threat(self) -> bool:
        """True if target is considered a threat (approaching and close)."""
        return self.threat_score > 30

    def position_at(self, delta_t: float) -> Tuple[float, float, float]:
        """
        Predict position at a future time using constant velocity model.

        Args:
            delta_t: Time offset from current state (seconds). Positive = future.

        Returns:
            (x, y, z) predicted position
        """
        return (
            self.x + self.vx * delta_t,
            self.y + self.vy * delta_t,
            self.z + self.vz * delta_t
        )

    def range_at(self, delta_t: float) -> float:
        """Predict range at a future time."""
        x, y, z = self.position_at(delta_t)
        return np.sqrt(x**2 + y**2 + z**2)


@dataclass
class StaticObject:
    """Confirmed static object in world map."""
    x: float
    y: float
    z: float
    radius: float  # estimated size
    detections: int  # total times detected
    first_seen: float
    last_seen: float
    object_id: int = 0


@dataclass
class RadarHealth:
    """Radar system health metrics."""
    fps: float = 0.0
    dropped_frames: int = 0
    bytes_per_sec: float = 0.0
    buffer_size: int = 0
    total_frames: int = 0
    active_tracks: int = 0
    static_objects: int = 0


@dataclass
class WorldState:
    """Complete world model, shared with renderer."""
    timestamp: float
    frame_number: int
    tracks: List[Track]
    static_objects: List[StaticObject]
    occupancy_grid: Optional[np.ndarray]  # 2D grid for ground plane
    radar_health: RadarHealth
    
    def get_tracks_by_classification(self, classification: ObjectClass) -> List[Track]:
        """Filter tracks by classification type."""
        return [t for t in self.tracks if t.classification == classification]
    
    def get_closest_track(self) -> Optional[Track]:
        """Get the closest tracked object."""
        if not self.tracks:
            return None
        return min(self.tracks, key=lambda t: t.range)

    # ===== Threat Assessment Methods =====

    def get_highest_threat(self) -> Optional[Track]:
        """Get the track with the highest threat score."""
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda t: t.threat_score)

    def get_threats(self, min_score: float = 30.0) -> List[Track]:
        """Get all tracks above a threat threshold, sorted by threat score."""
        threats = [t for t in self.tracks if t.threat_score >= min_score]
        return sorted(threats, key=lambda t: t.threat_score, reverse=True)

    def get_approaching_tracks(self) -> List[Track]:
        """Get all tracks that are approaching the radar."""
        return [t for t in self.tracks if t.is_approaching]

    def get_tracks_by_tti(self, max_tti: float = 5.0) -> List[Track]:
        """Get tracks sorted by time-to-intercept (soonest first)."""
        approaching = [t for t in self.tracks if t.time_to_intercept < max_tti]
        return sorted(approaching, key=lambda t: t.time_to_intercept)


@dataclass
class RawFrame:
    """Raw frame data from serial reader."""
    data: bytes
    timestamp: float
    frame_number: int

