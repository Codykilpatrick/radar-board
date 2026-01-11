"""
Configuration for the radar pipeline.

Loads settings from settings.yaml if available, otherwise uses defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path


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
    min_snr: float = 0.0  # dB - NOTE: many radars don't output SNR
    
    # Tracking parameters (tuned for fast acquisition / CIWS-style)
    association_gate: float = 0.75  # max distance for association (meters)
    min_confidence: float = 0.3  # minimum confidence to display
    min_hits: int = 1  # require N consecutive hits before showing track (1=instant)
    max_misses: int = 10  # frames before track deletion
    static_velocity_threshold: float = 0.1  # m/s
    static_frames_required: int = 20  # frames at low velocity to classify static

    # Track lifecycle (set initial >= min for instant acquisition)
    initial_confidence: float = 0.4  # start high for instant acquisition
    confidence_increment: float = 0.15  # confidence gained per hit
    confidence_decay: float = 0.08  # faster decay to clear ghost tracks
    
    # Kalman filter tuning
    process_noise: float = 0.5  # higher = trust measurements more
    measurement_noise: float = 0.05  # radar position uncertainty ~5cm
    frame_period: float = 0.05  # 20 Hz radar update rate
    
    # Debug output
    debug_tracking: bool = False  # print tracking diagnostics
    debug_detections: bool = False  # print raw detection info
    
    # Visualization
    trail_length: int = 50  # frames of history to show
    update_rate: int = 20  # Hz for render updates
    
    # Occupancy grid (optional)
    grid_enabled: bool = False
    grid_resolution: float = 0.1  # meters per cell
    grid_size: Tuple[int, int] = (100, 100)  # cells (10m x 10m)

    # Background subtraction
    bg_enabled: bool = True  # Enable background filtering if model exists
    bg_model_path: str = "background.npz"  # Path to background model file
    bg_resolution: float = 0.1  # Voxel size in meters (10cm default)
    bg_min_hits: int = 3  # Min detections per cell to mark as background
    bg_learning_duration: float = 10.0  # Seconds to learn when using --learn-background
    
    # Queue sizes
    raw_queue_size: int = 100
    frame_queue_size: int = 100
    state_queue_size: int = 10
    
    # Magic word for frame synchronization
    magic_word: bytes = field(default_factory=lambda: bytes([
        0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07
    ]))


def load_yaml_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to YAML file. If None, looks for settings.yaml
                   in the radar_pipeline directory.
    
    Returns:
        PipelineConfig with values from YAML (or defaults if file not found)
    """
    if yaml_path is None:
        # Look for settings.yaml in the same directory as this file
        yaml_path = Path(__file__).parent / "settings.yaml"
    else:
        yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        print(f"[Config] No settings.yaml found, using defaults")
        return PipelineConfig()
    
    try:
        import yaml
    except ImportError:
        print("[Config] PyYAML not installed, using defaults. Install with: pip install pyyaml")
        return PipelineConfig()
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return PipelineConfig()
        
        # Flatten nested structure into config
        config_dict = {}
        
        # Serial
        if 'serial' in data:
            config_dict['serial_port'] = data['serial'].get('port', PipelineConfig.serial_port)
            config_dict['baudrate'] = data['serial'].get('baudrate', PipelineConfig.baudrate)
        
        # Filtering
        if 'filtering' in data:
            config_dict['min_range'] = data['filtering'].get('min_range', PipelineConfig.min_range)
            config_dict['max_range'] = data['filtering'].get('max_range', PipelineConfig.max_range)
            config_dict['fov_angle'] = data['filtering'].get('fov_angle', PipelineConfig.fov_angle)
            config_dict['min_snr'] = data['filtering'].get('min_snr', PipelineConfig.min_snr)
        
        # Association
        if 'association' in data:
            config_dict['association_gate'] = data['association'].get('gate', PipelineConfig.association_gate)
        
        # Tracking
        if 'tracking' in data:
            config_dict['min_hits'] = data['tracking'].get('min_hits', PipelineConfig.min_hits)
            config_dict['min_confidence'] = data['tracking'].get('min_confidence', PipelineConfig.min_confidence)
            config_dict['initial_confidence'] = data['tracking'].get('initial_confidence', PipelineConfig.initial_confidence)
            config_dict['confidence_increment'] = data['tracking'].get('confidence_increment', PipelineConfig.confidence_increment)
            config_dict['confidence_decay'] = data['tracking'].get('confidence_decay', PipelineConfig.confidence_decay)
            config_dict['max_misses'] = data['tracking'].get('max_misses', PipelineConfig.max_misses)
            config_dict['static_velocity_threshold'] = data['tracking'].get('static_velocity_threshold', PipelineConfig.static_velocity_threshold)
            config_dict['static_frames_required'] = data['tracking'].get('static_frames_required', PipelineConfig.static_frames_required)
        
        # Kalman
        if 'kalman' in data:
            config_dict['process_noise'] = data['kalman'].get('process_noise', PipelineConfig.process_noise)
            config_dict['measurement_noise'] = data['kalman'].get('measurement_noise', PipelineConfig.measurement_noise)
            config_dict['frame_period'] = data['kalman'].get('frame_period', PipelineConfig.frame_period)
        
        # Visualization
        if 'visualization' in data:
            config_dict['update_rate'] = data['visualization'].get('update_rate', PipelineConfig.update_rate)
            config_dict['trail_length'] = data['visualization'].get('trail_length', PipelineConfig.trail_length)
        
        # Occupancy grid
        if 'occupancy_grid' in data:
            config_dict['grid_enabled'] = data['occupancy_grid'].get('enabled', PipelineConfig.grid_enabled)
            config_dict['grid_resolution'] = data['occupancy_grid'].get('resolution', PipelineConfig.grid_resolution)
            size = data['occupancy_grid'].get('size', list(PipelineConfig.grid_size))
            config_dict['grid_size'] = tuple(size) if isinstance(size, list) else size
        
        # Debug
        if 'debug' in data:
            config_dict['debug_tracking'] = data['debug'].get('tracking', PipelineConfig.debug_tracking)
            config_dict['debug_detections'] = data['debug'].get('detections', PipelineConfig.debug_detections)

        # Background subtraction
        if 'background' in data:
            config_dict['bg_enabled'] = data['background'].get('enabled', PipelineConfig.bg_enabled)
            config_dict['bg_model_path'] = data['background'].get('model_path', PipelineConfig.bg_model_path)
            config_dict['bg_resolution'] = data['background'].get('resolution', PipelineConfig.bg_resolution)
            config_dict['bg_min_hits'] = data['background'].get('min_hits', PipelineConfig.bg_min_hits)
            config_dict['bg_learning_duration'] = data['background'].get('learning_duration', PipelineConfig.bg_learning_duration)

        print(f"[Config] Loaded settings from {yaml_path}")
        return PipelineConfig(**config_dict)
        
    except Exception as e:
        print(f"[Config] Error loading {yaml_path}: {e}")
        return PipelineConfig()


def load_config(config_dict: dict) -> PipelineConfig:
    """Create a PipelineConfig from a dictionary, using defaults for missing keys."""
    return PipelineConfig(**{
        k: v for k, v in config_dict.items()
        if hasattr(PipelineConfig, k)
    })


# Default configuration - loads from YAML if available
DEFAULT_CONFIG = load_yaml_config()

