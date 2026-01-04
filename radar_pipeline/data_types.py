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


@dataclass
class RawFrame:
    """Raw frame data from serial reader."""
    data: bytes
    timestamp: float
    frame_number: int

