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
import queue
import time
import signal
import argparse
from typing import Optional

from pyqtgraph.Qt import QtWidgets, QtGui

from .config import PipelineConfig, DEFAULT_CONFIG
from .track_manager import TrackManager
from .render_engine import RenderEngine
from .data_source import DataSource, LiveDataSource, FileDataSource, SyntheticDataSource
from .recorder import RecordingDataSource


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
                 playback_speed: float = 1.0):
        """
        Initialize the radar pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
            source_type: Data source type ("live", "file", "synthetic")
            source_path: Path for file playback, or scenario name for synthetic
            record_path: If provided, record frames to this file
            playback_speed: Playback speed multiplier for file source
        """
        self.config = config or DEFAULT_CONFIG
        self.source_type = source_type
        self.source_path = source_path
        self.record_path = record_path
        self.playback_speed = playback_speed

        # Create queues
        self.frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self.state_queue = queue.Queue(maxsize=self.config.state_queue_size)

        # Components (created on start)
        self.data_source: Optional[DataSource] = None
        self.track_manager: Optional[TrackManager] = None
        self.render_engine: Optional[RenderEngine] = None

        self._running = False
    
    def _create_data_source(self) -> DataSource:
        """Create the appropriate data source based on configuration."""
        if self.source_type == "live":
            source = LiveDataSource(self.frame_queue, self.config)
        elif self.source_type == "file":
            if not self.source_path:
                raise ValueError("File source requires a path")
            source = FileDataSource(
                self.frame_queue,
                self.config,
                file_path=self.source_path,
                speed=self.playback_speed,
                loop=True
            )
        elif self.source_type == "synthetic":
            scenario = self.source_path or "forward"
            source = SyntheticDataSource(
                self.frame_queue,
                self.config,
                scenario=scenario
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
            config=self.config
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
  %(prog)s --source synthetic:multi --record test.jsonl  # Record synthetic
        """
    )

    parser.add_argument(
        '--source', '-s',
        default='live',
        help='Data source: live, file:<path>, synthetic:<scenario>'
    )
    parser.add_argument(
        '--record', '-r',
        metavar='PATH',
        help='Record frames to JSONL file'
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

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Create and run pipeline
    pipeline = RadarPipeline(
        source_type=source_type,
        source_path=source_path,
        record_path=args.record,
        playback_speed=args.speed
    )
    sys.exit(pipeline.start())


if __name__ == '__main__':
    main()

