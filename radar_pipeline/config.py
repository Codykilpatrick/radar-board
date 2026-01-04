"""
Configuration for the radar pipeline.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PipelineConfig:
    """Configuration parameters for the radar pipeline."""
    
    # Serial configuration
    serial_port: str = '/dev/tty.usbserial-00ED1D3E1'
    baudrate: int = 921600
    
    # Filtering parameters
    min_range: float = 0.25  # meters
    max_range: float = 8.0  # meters
    fov_angle: float = 60.0  # degrees (half angle, ±60° from center)
    min_snr: float = 0.0  # dB
    
    # Tracking parameters - TUNED FOR HUMAN TRACKING
    association_gate: float = 1.0  # max distance for association (meters) - larger for fast movement
    min_confidence: float = 0.2  # minimum confidence to display (show tracks faster)
    max_misses: int = 15  # frames before track deletion (coast longer through dropouts)
    static_velocity_threshold: float = 0.1  # m/s
    static_frames_required: int = 20  # frames at low velocity to classify static
    
    # Track lifecycle - TUNED FOR FASTER CONFIRMATION
    initial_confidence: float = 0.3  # start higher so tracks appear faster
    confidence_increment: float = 0.15  # build confidence faster
    confidence_decay: float = 0.03  # decay per missed frame (slow decay for coasting)
    
    # Kalman filter tuning - TUNED FOR HUMAN MOVEMENT
    process_noise: float = 0.5  # higher = trust measurements more, allow faster acceleration
    measurement_noise: float = 0.05  # radar position uncertainty ~5cm
    frame_period: float = 0.05  # 20 Hz radar update rate
    
    # Debug output
    debug_tracking: bool = False  # print tracking diagnostics
    
    # Visualization
    trail_length: int = 50  # frames of history to show
    update_rate: int = 20  # Hz for render updates
    
    # Occupancy grid (optional)
    grid_enabled: bool = True
    grid_resolution: float = 0.1  # meters per cell
    grid_size: Tuple[int, int] = (100, 100)  # cells (10m x 10m)
    
    # Queue sizes
    raw_queue_size: int = 100
    frame_queue_size: int = 100
    state_queue_size: int = 10
    
    # Magic word for frame synchronization
    magic_word: bytes = field(default_factory=lambda: bytes([
        0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07
    ]))


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()


def load_config(config_dict: dict) -> PipelineConfig:
    """Create a PipelineConfig from a dictionary, using defaults for missing keys."""
    return PipelineConfig(**{
        k: v for k, v in config_dict.items()
        if hasattr(PipelineConfig, k)
    })

