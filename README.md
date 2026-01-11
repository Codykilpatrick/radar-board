# Radar Board

Real-time 3D visualization and tracking system for TI IWR6843ISK mmWave radar on an RC airboat.

## Overview

This project provides tools for visualizing and tracking objects detected by a TI IWR6843ISK 60GHz mmWave radar. It includes both simple single-file visualizers and a full multi-threaded pipeline with Kalman filtering.

## Components

### Radar Pipeline (Recommended)

Multi-threaded pipeline with Kalman filtering, track management, and 3D visualization.

```bash
# Run with live radar
python3 -m radar_pipeline.main

# Run with synthetic data (no radar needed)
python3 -m radar_pipeline.main --source synthetic:crossing

# Playback a recorded session
python3 -m radar_pipeline.main --source file:recording.jsonl

# Record a session
python3 -m radar_pipeline.main --record session.jsonl
```

See [`radar_pipeline/README.md`](radar_pipeline/README.md) for full documentation.

### Legacy Visualizers

| File | Description |
|------|-------------|
| `visualize_radar_3d.py` | Simple 3D visualizer (single-threaded) |
| `visualize_radar_qt.py` | Qt-based 2D visualizer |

## Quick Start

### 1. Install Dependencies

```bash
pip install pyqtgraph pyserial numpy scipy PyQt6 PyOpenGL pyyaml opencv-python
```

### 2. Configure Radar

Choose a radar profile based on your use case:

| Profile | Max Velocity | Clutter Removal | Best For |
|---------|-------------|-----------------|----------|
| `profile_2d.cfg` | 3 m/s | ON | Close-range, slow targets |
| `profile_fast.cfg` | 9.7 m/s (22 mph) | ON | Running, jogging, approaching targets |
| `profile_nofilter.cfg` | 9.7 m/s (22 mph) | OFF | Parallel motion, static objects |

Send configuration to the radar:

```bash
python3 send_config.py profile_fast.cfg    # Recommended for most uses
python3 send_config.py profile_nofilter.cfg # For parallel motion detection
python3 send_config.py profile_2d.cfg       # Conservative/original
```

**Note:** FMCW radar measures radial velocity (toward/away). Targets moving parallel to the radar have near-zero radial velocity and may be filtered out with clutter removal ON.

### 3. Run Visualization

```bash
# Multi-threaded pipeline (recommended)
python3 -m radar_pipeline.main

# Or simple 3D visualizer
python3 visualize_radar_3d.py
```

## Hardware Setup

- **Radar**: TI IWR6843ISK
- **Connection**: USB serial at 921600 baud
- **Default Port**: `/dev/tty.usbserial-00ED1D3E1` (macOS)

## Recording & Playback

Record radar data and synchronized video for later analysis:

```bash
# Record a session (creates timestamped directory in recordings/)
python3 -m radar_pipeline.main --record

# Record with video from webcam
python3 -m radar_pipeline.main --record --record-video

# Playback a recorded session
python3 -m radar_pipeline.main --source file:recordings/session_20240115_143022/radar.jsonl

# Playback with synchronized video
python3 -m radar_pipeline.main --source file:recordings/session_20240115_143022/radar.jsonl \
    --play-video recordings/session_20240115_143022/video
```

Each recording session creates a directory with:
- `radar.jsonl` - Radar detections with timestamps
- `video.mp4` - Webcam video (if `--record-video` used)
- `video_timestamps.json` - Frame timestamps for sync

## Background Subtraction (Stationary Object Detection)

Detect stationary objects like hovering drones or standing people by learning and filtering out static environment clutter.

```bash
# Step 1: Learn the background (room should be empty of targets)
# Use profile_nofilter.cfg for stationary object detection
python3 send_config.py profile_nofilter.cfg
python3 -m radar_pipeline.main --learn-background 10

# Step 2: Run with background filtering enabled
python3 -m radar_pipeline.main

# Run without background filtering
python3 -m radar_pipeline.main --no-background

# Use a custom background model path
python3 -m radar_pipeline.main --bg-model /path/to/model.npz
```

The background model is saved to `background.npz` and automatically loaded on subsequent runs. Configure in `settings.yaml`:

```yaml
background:
  enabled: true
  model_path: "background.npz"
  resolution: 0.1       # 10cm voxel size
  min_hits: 3           # Detections per cell to mark as background
```

## Offline Development

Develop and test without physical radar hardware:

```bash
# List available synthetic scenarios
python3 -m radar_pipeline.main --list-scenarios

# Run with synthetic data
python3 -m radar_pipeline.main --source synthetic:multi      # Multiple targets
python3 -m radar_pipeline.main --source synthetic:dropout    # Test track coasting
python3 -m radar_pipeline.main --source synthetic:clutter    # Static + moving

# Record synthetic data for reproducible testing
python3 -m radar_pipeline.main --source synthetic:crossing --record

# Playback at different speeds
python3 -m radar_pipeline.main --source file:recordings/session_.../radar.jsonl --speed 0.5
python3 -m radar_pipeline.main --source file:recordings/session_.../radar.jsonl --speed 2.0
```

## Testing

```bash
# Run all unit tests
python3 radar_pipeline/tests/run_all_tests.py

# Visual tests with synthetic targets
python3 radar_pipeline/tests/visual_test.py --list           # List scenarios
python3 radar_pipeline/tests/visual_test.py crossing         # By name
python3 radar_pipeline/tests/visual_test.py 4                # Dropout/coasting test
```

## Project Structure

```
radar-board/
├── README.md                    # This file
├── profile_2d.cfg               # Radar config (conservative)
├── profile_fast.cfg             # Radar config (high speed)
├── profile_nofilter.cfg         # Radar config (no clutter filter)
├── send_config.py               # Send config to radar
├── read_radar.py                # Basic radar data reader
├── test_radar.py                # Radar connection test
├── visualize_radar_3d.py        # Simple 3D visualizer
├── visualize_radar_qt.py        # Simple 2D visualizer
├── recordings/                  # Recorded sessions (gitignored)
└── radar_pipeline/              # Multi-threaded pipeline
    ├── README.md                # Pipeline documentation
    ├── main.py                  # Entry point with CLI
    ├── config.py                # Configuration
    ├── settings.yaml            # Runtime configuration
    ├── data_source.py           # Data source abstraction
    ├── recorder.py              # JSONL recording
    ├── video_recorder.py        # Synchronized video recording
    ├── background.py            # Background subtraction model
    ├── scenarios.py             # Synthetic test scenarios
    ├── metrics.py               # Tracking performance metrics
    ├── serial_reader.py         # Serial reader thread
    ├── frame_parser.py          # Frame parser thread
    ├── track_manager.py         # Kalman tracking thread
    ├── render_engine.py         # 3D visualization
    ├── kalman.py                # Kalman filter
    ├── association.py           # Track-detection association
    └── tests/                   # Test suite
        ├── run_all_tests.py     # Run all tests
        ├── visual_test.py       # Visual test scenarios
        └── test_*.py            # Unit tests
```

## Features

- **3D Point Cloud Visualization**: Real-time display of radar detections
- **Kalman Filtering**: Smooth position and velocity estimation
- **Track Management**: Persistent object tracking with coasting through dropouts
- **Background Subtraction**: Learn static environment to detect new stationary objects
- **Threat Assessment**: Closing velocity, time-to-intercept, and threat scoring
- **Static/Dynamic Classification**: Automatic classification of stationary vs moving objects
- **Multi-threaded Architecture**: Serial reading doesn't block visualization
- **Synchronized Video Recording**: Record webcam alongside radar for ground truth
- **Offline Development**: Synthetic data generation and session recording/playback
- **Multiple Data Sources**: Live radar, recorded files, or synthetic patterns

## License

MIT

