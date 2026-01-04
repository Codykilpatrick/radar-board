#!/usr/bin/env python3
"""
Radar Pipeline - Main entry point.

Multi-threaded radar data processing pipeline for TI IWR6843ISK.

Architecture:
    SerialReader → FrameParser → TrackManager → RenderEngine
"""

import sys
import queue
import time
import signal
from typing import Optional

from pyqtgraph.Qt import QtWidgets, QtGui

from .config import PipelineConfig, DEFAULT_CONFIG
from .serial_reader import SerialReader
from .frame_parser import FrameParser
from .track_manager import TrackManager
from .render_engine import RenderEngine


class RadarPipeline:
    """
    Main radar pipeline orchestrator.
    
    Creates and manages all pipeline components:
    - SerialReader (thread)
    - FrameParser (thread)
    - TrackManager (thread)
    - RenderEngine (Qt main thread)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the radar pipeline.
        
        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or DEFAULT_CONFIG
        
        # Create queues
        self.raw_queue = queue.Queue(maxsize=self.config.raw_queue_size)
        self.frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self.state_queue = queue.Queue(maxsize=self.config.state_queue_size)
        
        # Components (created on start)
        self.serial_reader: Optional[SerialReader] = None
        self.frame_parser: Optional[FrameParser] = None
        self.track_manager: Optional[TrackManager] = None
        self.render_engine: Optional[RenderEngine] = None
        
        self._running = False
    
    def start(self):
        """
        Start the radar pipeline.
        
        Creates and starts all worker threads, then runs the Qt event loop.
        """
        print("=" * 60)
        print("IWR6843ISK Radar Pipeline")
        print("=" * 60)
        print(f"Serial: {self.config.serial_port} @ {self.config.baudrate}")
        print(f"Range: {self.config.min_range}m - {self.config.max_range}m")
        print(f"FOV: ±{self.config.fov_angle}°")
        print(f"Update Rate: {self.config.update_rate} Hz")
        print("=" * 60)
        print("Controls: Left-drag to rotate, Right-drag to zoom, Middle-drag to pan")
        print("=" * 60)
        
        # Create worker threads
        self.serial_reader = SerialReader(
            port=self.config.serial_port,
            baudrate=self.config.baudrate,
            output_queue=self.raw_queue,
            config=self.config
        )
        
        self.frame_parser = FrameParser(
            input_queue=self.raw_queue,
            output_queue=self.frame_queue,
            config=self.config
        )
        
        self.track_manager = TrackManager(
            input_queue=self.frame_queue,
            state_queue=self.state_queue,
            config=self.config
        )
        
        # Start worker threads
        self.serial_reader.start()
        self.frame_parser.start()
        self.track_manager.start()
        
        self._running = True
        print("[Pipeline] Worker threads started")
        
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
        if self.serial_reader:
            self.serial_reader.stop()
        if self.frame_parser:
            self.frame_parser.stop()
        if self.track_manager:
            self.track_manager.stop()
        
        # Wait for threads to finish
        if self.serial_reader and self.serial_reader.is_alive():
            self.serial_reader.join(timeout=1.0)
        if self.frame_parser and self.frame_parser.is_alive():
            self.frame_parser.join(timeout=1.0)
        if self.track_manager and self.track_manager.is_alive():
            self.track_manager.join(timeout=1.0)
        
        print("[Pipeline] All threads stopped")
        print("Radar pipeline stopped.")
    
    def get_stats(self) -> dict:
        """Get pipeline statistics from all components."""
        stats = {}
        
        if self.serial_reader:
            stats['serial'] = self.serial_reader.get_health_stats()
        if self.frame_parser:
            stats['parser'] = self.frame_parser.get_stats()
        if self.track_manager:
            stats['tracker'] = self.track_manager.get_stats()
        
        stats['queues'] = {
            'raw_queue': self.raw_queue.qsize(),
            'frame_queue': self.frame_queue.qsize(),
            'state_queue': self.state_queue.qsize(),
        }
        
        return stats


def main():
    """Entry point for the radar pipeline."""
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Create and run pipeline with default config
    pipeline = RadarPipeline()
    sys.exit(pipeline.start())


if __name__ == '__main__':
    main()

