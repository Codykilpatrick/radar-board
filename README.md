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
pip install pyqtgraph pyserial numpy scipy PyQt6 PyOpenGL
```

### 2. Configure Radar

Send configuration to the radar:

```bash
python3 send_config.py profile_2d.cfg
```

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

## Testing

```bash
# Run all unit tests
python3 radar_pipeline/tests/run_all_tests.py

# Visual test with synthetic targets
python3 radar_pipeline/tests/visual_test.py 4  # Dropout/coasting test
```

## Project Structure

```
radar-board/
├── README.md                    # This file
├── profile_2d.cfg               # Radar configuration
├── send_config.py               # Send config to radar
├── read_radar.py                # Basic radar data reader
├── test_radar.py                # Radar connection test
├── visualize_radar_3d.py        # Simple 3D visualizer
├── visualize_radar_qt.py        # Simple 2D visualizer
└── radar_pipeline/              # Multi-threaded pipeline
    ├── README.md                # Pipeline documentation
    ├── main.py                  # Entry point
    ├── config.py                # Configuration
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
- **Static/Dynamic Classification**: Automatic classification of stationary vs moving objects
- **Multi-threaded Architecture**: Serial reading doesn't block visualization

## License

MIT

