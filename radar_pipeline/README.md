# Radar Pipeline

Multi-threaded radar data processing pipeline for TI IWR6843ISK mmWave radar.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Serial    │    │   Frame     │    │   Track     │    │   Render    │
│   Reader    │───▶│   Parser    │───▶│   Manager   │───▶│   Engine    │
│  (Thread)   │    │  (Thread)   │    │  (Thread)   │    │ (Main/Qt)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  raw_queue         frame_queue        state_queue        WorldState
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| **SerialReader** | `serial_reader.py` | Reads raw bytes from radar, finds frame boundaries |
| **FrameParser** | `frame_parser.py` | Parses TLV structure, extracts detections, applies filtering |
| **TrackManager** | `track_manager.py` | Kalman filtering, track association, static/dynamic classification |
| **RenderEngine** | `render_engine.py` | Qt/OpenGL 3D visualization |

### Supporting Modules

| Module | Description |
|--------|-------------|
| `kalman.py` | 6-state Kalman filter (x, y, z, vx, vy, vz) |
| `association.py` | Hungarian/greedy track-detection association with Mahalanobis distance |
| `data_types.py` | Detection, Track, WorldState dataclasses |
| `config.py` | Configuration with tunable parameters |
| `utils.py` | Coordinate transforms and helpers |

## Installation

Requires Python 3.8+ with the following packages:

```bash
pip install pyqtgraph pyserial numpy scipy PyQt6 PyOpenGL
```

## Usage

### Run with Real Radar

```bash
cd /path/to/radar-board
python3 -m radar_pipeline.main
```

### Run with Debug Output

```python
from radar_pipeline import RadarPipeline, PipelineConfig

# Enable debug output to see raw detection counts
config = PipelineConfig(debug_tracking=True, debug_detections=True)
pipeline = RadarPipeline(config)
pipeline.start()
```

### Custom Configuration

```python
from radar_pipeline import RadarPipeline, PipelineConfig

config = PipelineConfig(
    serial_port='/dev/tty.usbserial-00ED1D3E1',
    max_range=10.0,           # Increase detection range
    association_gate=1.5,     # Larger gate for fast movement
    min_confidence=0.2,       # Show tracks faster
    debug_tracking=True,      # Print tracking diagnostics
)

pipeline = RadarPipeline(config)
pipeline.start()
```

## Configuration Parameters

### Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_range` | 0.25m | Ignore detections closer than this |
| `max_range` | 8.0m | Maximum detection range |
| `fov_angle` | 60° | Half-angle of field of view (±60°) |
| `min_snr` | 0.0 dB | Minimum SNR threshold |

### Tracking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `association_gate` | 1.0m | Max Mahalanobis distance for association |
| `min_confidence` | 0.2 | Minimum confidence to display track |
| `max_misses` | 15 | Frames before track deletion |
| `initial_confidence` | 0.3 | Starting confidence for new tracks |
| `confidence_increment` | 0.15 | Confidence increase per hit |
| `confidence_decay` | 0.03 | Confidence decrease per miss |

### Kalman Filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `process_noise` | 0.5 | Higher = trust measurements more |
| `measurement_noise` | 0.05 | Radar position uncertainty (~5cm) |
| `frame_period` | 0.05s | Expected time between frames (20 Hz) |

## Visualization Legend

### Track Icons (Info Panel)

| Icon | Color | Meaning |
|------|-------|---------|
| ● | Cyan | Active track receiving detections |
| ◌ | Magenta | Coasting track (predicted position) |
| ◆ | Gray | Static object |

### Track Status

| Status | Meaning |
|--------|---------|
| ✓N | N consecutive hits (receiving detections) |
| ~N | N consecutive misses (coasting) |

### 3D View Colors

| Color | Meaning |
|-------|---------|
| Red | Object < 1m away |
| Yellow | Object 1-3m away |
| Green | Object > 3m away |
| Magenta (fading) | Coasting (no detection) |
| Gray | Static object |

## Testing

### Run All Unit Tests

```bash
cd /path/to/radar-board
python3 radar_pipeline/tests/run_all_tests.py
```

### Run Individual Test Suites

```bash
# Kalman filter tests
python3 radar_pipeline/tests/test_kalman.py

# Association tests
python3 radar_pipeline/tests/test_association.py

# Tracking tests
python3 radar_pipeline/tests/test_tracking.py

# Simulation tests
python3 radar_pipeline/tests/test_simulation.py
```

### Visual Tests

Visual tests display synthetic moving targets in the 3D visualizer to validate tracking behavior.

```bash
# Scenario 1: Single target moving forward/back
python3 radar_pipeline/tests/visual_test.py 1

# Scenario 2: Circular motion
python3 radar_pipeline/tests/visual_test.py 2

# Scenario 3: Two targets crossing paths
python3 radar_pipeline/tests/visual_test.py 3

# Scenario 4: Dropout test (coasting)
python3 radar_pipeline/tests/visual_test.py 4

# Scenario 5: Vertical motion (Z test)
python3 radar_pipeline/tests/visual_test.py 5

# Scenario 6: Random walk stress test
python3 radar_pipeline/tests/visual_test.py 6
```

#### What to Look For

| Scenario | Expected Behavior |
|----------|-------------------|
| **1 - Forward/Back** | Single dot moving smoothly, trail follows |
| **2 - Circular** | Target orbits with smooth tracking |
| **3 - Crossing** | Two separate tracks, no confusion at crossing |
| **4 - Dropout** | Track turns magenta during dropout, recovers |
| **5 - Vertical** | Target visibly moves up/down (Z working) |
| **6 - Random** | Track follows erratic motion, stays locked |

## Troubleshooting

### Tracks Keep Dropping

1. **Enable debug output** to see raw detection counts:
   ```python
   config = PipelineConfig(debug_tracking=True, debug_detections=True)
   ```

2. **Check raw detections**: If `Raw=0`, the radar isn't seeing you (check position, clothing)

3. **Loosen tracking parameters**:
   ```python
   config = PipelineConfig(
       association_gate=1.5,    # Larger gate
       min_confidence=0.1,      # Show earlier
       max_misses=20,           # Coast longer
   )
   ```

### Tracks Jitter/Jump

1. **Increase measurement noise** to trust predictions more:
   ```python
   config = PipelineConfig(measurement_noise=0.1)
   ```

2. **Decrease process noise** to smooth motion:
   ```python
   config = PipelineConfig(process_noise=0.2)
   ```

### Z Values Always Zero

Your radar config may be 2D only. Check your `.cfg` file for proper TX antenna configuration for elevation detection.

## File Structure

```
radar_pipeline/
├── __init__.py          # Package exports
├── main.py              # RadarPipeline entry point
├── config.py            # Configuration dataclass
├── data_types.py        # Detection, Track, WorldState
├── serial_reader.py     # Serial reader thread
├── frame_parser.py      # Frame parser thread
├── track_manager.py     # Track manager thread
├── render_engine.py     # Qt/OpenGL visualization
├── kalman.py            # Kalman filter
├── association.py       # Track-detection association
├── utils.py             # Utilities
├── README.md            # This file
└── tests/
    ├── __init__.py
    ├── run_all_tests.py     # Run all tests
    ├── test_kalman.py       # Kalman filter tests
    ├── test_association.py  # Association tests
    ├── test_tracking.py     # Tracking tests
    ├── test_simulation.py   # Simulation tests
    └── visual_test.py       # Visual test scenarios
```

