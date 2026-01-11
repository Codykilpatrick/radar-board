#!/usr/bin/env python3
"""
Radar Pipeline - Main entry point.

Multi-threaded radar data processing pipeline for TI IWR6843ISK.

Architecture:
    DataSource → TrackManager → RenderEngine

Data sources:
    - live: Real radar via serial port (default)
    - file:<path>: Playback from JSONL recording
    - synthetic:<scenario>: Generated test patterns
"""

import sys
import os
import queue
import time
import signal
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyqtgraph.Qt import QtWidgets, QtGui

from .config import PipelineConfig, DEFAULT_CONFIG
from .track_manager import TrackManager
from .render_engine import RenderEngine
from .data_source import DataSource, LiveDataSource, FileDataSource, SyntheticDataSource
from .recorder import RecordingDataSource
from .video_recorder import VideoRecorder, VideoPlayer, check_camera_available
from .background import BackgroundModel


class RadarPipeline:
    """
    Main radar pipeline orchestrator.

    Creates and manages all pipeline components:
    - DataSource (thread) - live, file, or synthetic
    - TrackManager (thread)
    - RenderEngine (Qt main thread)
    """

    def __init__(self,
                 config: Optional[PipelineConfig] = None,
                 source_type: str = "live",
                 source_path: Optional[str] = None,
                 record_path: Optional[str] = None,
                 playback_speed: float = 1.0,
                 record_video: Optional[str] = None,
                 play_video: Optional[str] = None,
                 background_model: Optional[BackgroundModel] = None):
        """
        Initialize the radar pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
            source_type: Data source type ("live", "file", "synthetic")
            source_path: Path for file playback, or scenario name for synthetic
            record_path: If provided, record frames to this file
            playback_speed: Playback speed multiplier for file source
            record_video: If provided, record video to this path (without extension)
            play_video: If provided, play synchronized video from this path
            background_model: Optional background model for clutter filtering
        """
        self.config = config or DEFAULT_CONFIG
        self.source_type = source_type
        self.source_path = source_path
        self.record_path = record_path
        self.playback_speed = playback_speed
        self.record_video_path = record_video
        self.play_video_path = play_video
        self.background_model = background_model

        # Create queues
        self.frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self.state_queue = queue.Queue(maxsize=self.config.state_queue_size)

        # Components (created on start)
        self.data_source: Optional[DataSource] = None
        self.track_manager: Optional[TrackManager] = None
        self.render_engine: Optional[RenderEngine] = None
        self.video_recorder: Optional[VideoRecorder] = None
        self.video_player: Optional[VideoPlayer] = None

        self._running = False
    
    def _create_data_source(self) -> DataSource:
        """Create the appropriate data source based on configuration."""
        if self.source_type == "live":
            source = LiveDataSource(
                self.frame_queue,
                self.config,
                background_model=self.background_model
            )
        elif self.source_type == "file":
            if not self.source_path:
                raise ValueError("File source requires a path")
            source = FileDataSource(
                self.frame_queue,
                self.config,
                file_path=self.source_path,
                speed=self.playback_speed,
                loop=True,
                background_model=self.background_model
            )
        elif self.source_type == "synthetic":
            scenario = self.source_path or "forward"
            source = SyntheticDataSource(
                self.frame_queue,
                self.config,
                scenario=scenario,
                background_model=self.background_model
            )
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")

        # Wrap with recorder if recording is enabled
        if self.record_path:
            source = RecordingDataSource(source, self.frame_queue, self.record_path)

        return source

    def start(self):
        """
        Start the radar pipeline.

        Creates and starts all worker threads, then runs the Qt event loop.
        """
        print("=" * 60)
        print("IWR6843ISK Radar Pipeline")
        print("=" * 60)

        # Print source info
        if self.source_type == "live":
            print(f"Source: Live radar ({self.config.serial_port} @ {self.config.baudrate})")
        elif self.source_type == "file":
            print(f"Source: File playback ({self.source_path}) at {self.playback_speed}x")
        elif self.source_type == "synthetic":
            print(f"Source: Synthetic ({self.source_path or 'forward'})")

        if self.record_path:
            print(f"Recording to: {self.record_path}")
        if self.record_video_path:
            print(f"Recording video to: {self.record_video_path}.mp4")
        if self.play_video_path:
            print(f"Playing video from: {self.play_video_path}")
        if self.background_model:
            stats = self.background_model.get_stats()
            print(f"Background filter: {stats['background_cells']} cells ({stats['background_pct']:.1f}% of volume)")

        print(f"Range: {self.config.min_range}m - {self.config.max_range}m")
        print(f"FOV: ±{self.config.fov_angle}°")
        print(f"Update Rate: {self.config.update_rate} Hz")
        print("=" * 60)
        print("Controls: Left-drag to rotate, Right-drag to zoom, Middle-drag to pan")
        print("=" * 60)

        # Create data source
        self.data_source = self._create_data_source()

        # Create track manager
        self.track_manager = TrackManager(
            input_queue=self.frame_queue,
            state_queue=self.state_queue,
            config=self.config
        )

        # Start workers
        self.data_source.start()
        self.track_manager.start()

        # Start video recorder if requested
        if self.record_video_path:
            if check_camera_available():
                self.video_recorder = VideoRecorder(self.record_video_path)
                if not self.video_recorder.start():
                    print("[Pipeline] Warning: Failed to start video recording")
                    self.video_recorder = None
            else:
                print("[Pipeline] Warning: No camera available for video recording")

        # Initialize video player if requested
        if self.play_video_path:
            self.video_player = VideoPlayer(self.play_video_path)
            if not self.video_player.start():
                print("[Pipeline] Warning: Failed to load video for playback")
                self.video_player = None

        self._running = True
        print("[Pipeline] Workers started")
        
        # Create Qt application and render engine (main thread)
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Set dark palette
        palette = app.palette()
        palette.setColor(palette.ColorRole.Window, QtGui.QColor('#0a0a12'))
        palette.setColor(palette.ColorRole.WindowText, QtGui.QColor('#e0e0e0'))
        palette.setColor(palette.ColorRole.Base, QtGui.QColor('#141420'))
        palette.setColor(palette.ColorRole.Text, QtGui.QColor('#e0e0e0'))
        app.setPalette(palette)
        
        self.render_engine = RenderEngine(
            state_queue=self.state_queue,
            config=self.config,
            video_player=self.video_player
        )
        self.render_engine.show()
        
        # Connect close event to pipeline stop
        app.aboutToQuit.connect(self.stop)
        
        # Run Qt event loop
        exit_code = app.exec()
        
        # Cleanup
        self.stop()
        
        return exit_code
    
    def stop(self):
        """
        Stop the radar pipeline.

        Signals all threads to stop and waits for them to join.
        """
        if not self._running:
            return

        print("\n[Pipeline] Stopping...")
        self._running = False

        # Stop video recorder/player
        if self.video_recorder:
            self.video_recorder.stop()
            self.video_recorder = None
        if self.video_player:
            self.video_player.stop()
            self.video_player = None

        # Signal threads to stop
        if self.data_source:
            self.data_source.stop()
        if self.track_manager:
            self.track_manager.stop()

        # Wait for threads to finish
        if self.data_source and self.data_source.is_alive():
            self.data_source.join(timeout=1.0)
        if self.track_manager and self.track_manager.is_alive():
            self.track_manager.join(timeout=1.0)

        print("[Pipeline] All threads stopped")
        print("Radar pipeline stopped.")
    
    def get_stats(self) -> dict:
        """Get pipeline statistics from all components."""
        stats = {}

        if self.data_source and hasattr(self.data_source, 'get_stats'):
            stats['source'] = self.data_source.get_stats()
        if self.track_manager:
            stats['tracker'] = self.track_manager.get_stats()

        stats['queues'] = {
            'frame_queue': self.frame_queue.qsize(),
            'state_queue': self.state_queue.qsize(),
        }

        return stats


def create_session_directory(base_path: str = "recordings") -> Path:
    """
    Create a timestamped session directory for recording.

    Returns:
        Path to the created session directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base_path) / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def parse_source(source_str: str) -> tuple:
    """
    Parse source string into (type, path).

    Examples:
        "live" -> ("live", None)
        "file:recording.jsonl" -> ("file", "recording.jsonl")
        "synthetic:crossing" -> ("synthetic", "crossing")
    """
    if source_str == "live":
        return ("live", None)
    elif source_str.startswith("file:"):
        return ("file", source_str[5:])
    elif source_str.startswith("synthetic:"):
        return ("synthetic", source_str[10:])
    else:
        # Assume it's a file path if it exists or ends with .jsonl
        if source_str.endswith('.jsonl'):
            return ("file", source_str)
        return ("live", None)


def learn_background_mode(source_type: str,
                          source_path: Optional[str],
                          duration: float,
                          model_path: str) -> int:
    """
    Run background learning mode.

    Collects detections for the specified duration, builds a background model,
    and saves it to disk.

    Args:
        source_type: Data source type
        source_path: Source path (for file/synthetic)
        duration: Learning duration in seconds
        model_path: Path to save the model

    Returns:
        Exit code (0 = success)
    """
    print("=" * 60)
    print("Background Learning Mode")
    print("=" * 60)
    print(f"Duration: {duration} seconds")
    print(f"Output: {model_path}")
    print("=" * 60)
    print("Make sure the scene contains ONLY static objects (walls, floor, furniture)")
    print("Do NOT have any people or objects you want to detect in view")
    print("=" * 60)

    # Create background model
    config = DEFAULT_CONFIG
    model = BackgroundModel(
        resolution=config.bg_resolution,
        x_range=(-config.max_range, config.max_range),
        y_range=(0, config.max_range),
        z_range=(-2.0, 3.0)  # -2m to +3m height
    )

    # Create queue and data source (no background filtering during learning)
    frame_queue = queue.Queue(maxsize=config.frame_queue_size)

    if source_type == "live":
        source = LiveDataSource(frame_queue, config)
    elif source_type == "file":
        source = FileDataSource(
            frame_queue, config,
            file_path=source_path,
            speed=1.0,
            loop=False
        )
    elif source_type == "synthetic":
        source = SyntheticDataSource(
            frame_queue, config,
            scenario=source_path or "forward"
        )
    else:
        print(f"Error: Unknown source type '{source_type}'")
        return 1

    # Start data source
    source.start()
    print(f"\n[Learning] Started collecting detections...")

    start_time = time.time()
    frame_count = 0
    detection_count = 0

    try:
        while time.time() - start_time < duration:
            try:
                frame_num, timestamp, detections = frame_queue.get(timeout=0.1)
                frame_count += 1

                # Add all detections to background model
                for det in detections:
                    model.add_detection(det.x, det.y, det.z)
                    detection_count += 1

                # Progress update every 50 frames
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    print(f"[Learning] {elapsed:.1f}s elapsed, {remaining:.1f}s remaining, "
                          f"{frame_count} frames, {detection_count} detections")

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n[Learning] Interrupted by user")

    finally:
        source.stop()

    # Finalize and save
    print(f"\n[Learning] Collected {detection_count} detections from {frame_count} frames")

    if detection_count == 0:
        print("[Learning] Error: No detections collected. Check radar connection.")
        return 1

    model.finalize(min_hits=config.bg_min_hits)
    model.save(model_path)

    print("\n" + "=" * 60)
    print("Background learning complete!")
    print(f"Model saved to: {model_path}")
    print("Run normally to use background filtering:")
    print(f"  python3 -m radar_pipeline.main")
    print("=" * 60)

    return 0


def main():
    """Entry point for the radar pipeline."""
    parser = argparse.ArgumentParser(
        description="Radar Pipeline - Real-time tracking and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Live radar (default)
  %(prog)s --source synthetic:crossing  # Synthetic test pattern
  %(prog)s --source file:recording.jsonl # Playback recording
  %(prog)s --record session.jsonl       # Record live session

Background Learning (for stationary object detection):
  %(prog)s --learn-background 10        # Learn background for 10 seconds
  %(prog)s                              # Run with background filtering
  %(prog)s --no-background              # Run without background filtering
        """
    )

    parser.add_argument(
        '--source', '-s',
        default='live',
        help='Data source: live, file:<path>, synthetic:<scenario>'
    )
    parser.add_argument(
        '--record', '-r',
        nargs='?',
        const='auto',
        metavar='PATH',
        help='Record session. Creates timestamped dir in recordings/ if no path given.'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Playback speed multiplier for file source (default: 1.0)'
    )
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available synthetic scenarios and exit'
    )
    parser.add_argument(
        '--record-video',
        action='store_true',
        help='Record video from webcam alongside radar data'
    )
    parser.add_argument(
        '--play-video',
        metavar='PATH',
        help='Play synchronized video during playback (path without extension or .mp4)'
    )
    parser.add_argument(
        '--learn-background',
        type=float,
        metavar='SECONDS',
        help='Learn background for N seconds, save model, and exit'
    )
    parser.add_argument(
        '--no-background',
        action='store_true',
        help='Disable background filtering even if model exists'
    )
    parser.add_argument(
        '--bg-model',
        metavar='PATH',
        help='Path to background model file (default: background.npz)'
    )

    args = parser.parse_args()

    # Handle --list-scenarios
    if args.list_scenarios:
        from .scenarios import list_scenarios
        print("Available synthetic scenarios:")
        print("-" * 50)
        for name, desc in list_scenarios().items():
            print(f"  {name:12} - {desc}")
        print("-" * 50)
        print("Usage: --source synthetic:<name>")
        return 0

    # Parse source
    source_type, source_path = parse_source(args.source)

    # Handle recording paths
    record_path = None
    record_video_path = None

    if args.record:
        if args.record == 'auto':
            # Create timestamped session directory
            session_dir = create_session_directory()
            record_path = str(session_dir / "radar.jsonl")
            if args.record_video:
                record_video_path = str(session_dir / "video")
            print(f"[Session] Recording to {session_dir}/")
        elif args.record.endswith('.jsonl'):
            # Explicit path to JSONL file
            record_path = args.record
            if args.record_video:
                # Put video alongside the JSONL file
                record_video_path = args.record.replace('.jsonl', '_video')
        else:
            # Treat as directory, create if needed
            session_dir = Path(args.record)
            session_dir.mkdir(parents=True, exist_ok=True)
            record_path = str(session_dir / "radar.jsonl")
            if args.record_video:
                record_video_path = str(session_dir / "video")
    elif args.record_video:
        # Video recording without radar recording - create session dir
        session_dir = create_session_directory()
        record_video_path = str(session_dir / "video")
        print(f"[Session] Recording video to {session_dir}/")

    # Determine background model path
    bg_model_path = args.bg_model or DEFAULT_CONFIG.bg_model_path

    # Handle --learn-background mode
    if args.learn_background:
        return learn_background_mode(
            source_type=source_type,
            source_path=source_path,
            duration=args.learn_background,
            model_path=bg_model_path
        )

    # Load background model for normal operation
    background_model = None
    if not args.no_background and DEFAULT_CONFIG.bg_enabled:
        if BackgroundModel.exists(bg_model_path):
            try:
                background_model = BackgroundModel.load(bg_model_path)
            except Exception as e:
                print(f"[Background] Warning: Failed to load model: {e}")

    # Create and run pipeline
    pipeline = RadarPipeline(
        source_type=source_type,
        source_path=source_path,
        record_path=record_path,
        playback_speed=args.speed,
        record_video=record_video_path,
        play_video=args.play_video,
        background_model=background_model
    )
    sys.exit(pipeline.start())


if __name__ == '__main__':
    main()

