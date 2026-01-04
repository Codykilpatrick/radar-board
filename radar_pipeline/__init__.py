"""
Radar Pipeline - Multi-threaded radar data processing system

Architecture:
    SerialReader → FrameParser → TrackManager → RenderEngine
         │              │              │              │
         ▼              ▼              ▼              ▼
     raw_queue     frame_queue    state_queue    WorldState
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
from .main import RadarPipeline

__version__ = "1.0.0"
__all__ = [
    "ObjectClass",
    "Detection",
    "Track",
    "StaticObject",
    "WorldState",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    "SerialReader",
    "FrameParser",
    "TrackManager",
    "RenderEngine",
    "RadarPipeline",
]

