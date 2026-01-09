"""
Radar Pipeline - Multi-threaded radar data processing system

Architecture:
    DataSource → TrackManager → RenderEngine
         │              │              │
         ▼              ▼              ▼
    frame_queue    state_queue    WorldState

Data Sources:
    - LiveDataSource: Real radar via serial port
    - FileDataSource: Playback from JSONL recordings
    - SyntheticDataSource: Generated test patterns
"""

from .data_types import (
    ObjectClass,
    Detection,
    Track,
    StaticObject,
    WorldState,
)
from .config import PipelineConfig, DEFAULT_CONFIG
from .serial_reader import SerialReader
from .frame_parser import FrameParser
from .track_manager import TrackManager
from .render_engine import RenderEngine
from .data_source import (
    DataSource,
    LiveDataSource,
    FileDataSource,
    SyntheticDataSource,
)
from .recorder import FrameRecorder, RecordingDataSource
from .scenarios import SCENARIOS, list_scenarios
from .main import RadarPipeline

__version__ = "1.1.0"
__all__ = [
    # Data types
    "ObjectClass",
    "Detection",
    "Track",
    "StaticObject",
    "WorldState",
    # Config
    "PipelineConfig",
    "DEFAULT_CONFIG",
    # Pipeline components
    "SerialReader",
    "FrameParser",
    "TrackManager",
    "RenderEngine",
    # Data sources
    "DataSource",
    "LiveDataSource",
    "FileDataSource",
    "SyntheticDataSource",
    # Recording
    "FrameRecorder",
    "RecordingDataSource",
    # Scenarios
    "SCENARIOS",
    "list_scenarios",
    # Main
    "RadarPipeline",
]

