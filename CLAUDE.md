# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time 3D visualization and tracking system for TI IWR6843ISK mmWave radar. The main component is a multi-threaded pipeline (`radar_pipeline/`) with Kalman filtering and track management.

## Common Commands

```bash
# Run the main radar pipeline (requires connected radar)
python3 -m radar_pipeline.main

# Run with synthetic data (no radar needed)
python3 -m radar_pipeline.main --source synthetic:crossing
python3 -m radar_pipeline.main --source synthetic:multi

# Playback recorded data
python3 -m radar_pipeline.main --source file:recording.jsonl
python3 -m radar_pipeline.main --source file:recording.jsonl --speed 2.0

# Record a session (works with any source)
python3 -m radar_pipeline.main --record session.jsonl
python3 -m radar_pipeline.main --source synthetic:multi --record test.jsonl

# List available synthetic scenarios
python3 -m radar_pipeline.main --list-scenarios

# Run all unit tests
python3 radar_pipeline/tests/run_all_tests.py

# Visual tests with synthetic targets
python3 radar_pipeline/tests/visual_test.py 1        # Single target forward/back
python3 radar_pipeline/tests/visual_test.py crossing # By name
python3 radar_pipeline/tests/visual_test.py --list   # List scenarios

# Send configuration to radar hardware
python3 send_config.py profile_2d.cfg
```

## Architecture

The pipeline uses a producer-consumer pattern with threads connected by queues:

```
DataSource → TrackManager → RenderEngine
     │              │              │
     ▼              ▼              ▼
frame_queue    state_queue    WorldState
```

**Data Sources** (`data_source.py`):
- `LiveDataSource`: Real radar via SerialReader + FrameParser
- `FileDataSource`: Playback from JSONL recordings (loops at EOF)
- `SyntheticDataSource`: Generated test patterns from `scenarios.py`

**Core Components**:
- **TrackManager** (`track_manager.py`): Kalman filtering, Hungarian algorithm association, track lifecycle
- **RenderEngine** (`render_engine.py`): Qt/OpenGL 3D visualization on main thread

**Recording** (`recorder.py`):
- `FrameRecorder`: Writes detections to JSONL format
- `RecordingDataSource`: Wraps any source to record while running

Supporting modules:
- `kalman.py`: 6-state Kalman filter (x, y, z, vx, vy, vz)
- `association.py`: Track-detection association using Mahalanobis distance
- `scenarios.py`: Synthetic test patterns (forward, crossing, dropout, multi, etc.)
- `data_types.py`: Detection, Track, WorldState dataclasses

## Configuration

Runtime parameters are in `radar_pipeline/settings.yaml`. Key tuning areas:
- `filtering`: min/max range, FOV angle
- `tracking`: min_hits for confirmation, confidence thresholds, max_misses for coasting
- `kalman`: process_noise (trust measurements) vs measurement_noise (trust predictions)

To override programmatically:
```python
from radar_pipeline import RadarPipeline, PipelineConfig
config = PipelineConfig(debug_tracking=True, association_gate=1.0)
pipeline = RadarPipeline(config)
```

## Hardware

- Radar: TI IWR6843ISK at 921600 baud
- Default serial port: `/dev/tty.usbserial-00ED1D3E1` (macOS)
- Frame sync magic word: `0x02 0x01 0x04 0x03 0x06 0x05 0x08 0x07`

## Dependencies

```bash
pip install pyqtgraph pyserial numpy scipy PyQt6 PyOpenGL pyyaml
```
