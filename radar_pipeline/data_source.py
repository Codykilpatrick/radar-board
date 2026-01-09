"""
Data Source Abstraction - Unified interface for radar data input.

Provides a common interface for:
- LiveDataSource: Real radar via serial port
- FileDataSource: Playback from recorded JSONL files
- SyntheticDataSource: Generated test patterns
"""

import threading
import queue
import time
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from pathlib import Path

from .data_types import Detection
from .config import PipelineConfig


# Frame type: (frame_number, timestamp, List[Detection])
FrameData = Tuple[int, float, List[Detection]]


class DataSource(ABC):
    """
    Abstract base class for radar data sources.

    All data sources produce frames in the format:
        (frame_number, timestamp, List[Detection])

    Subclasses must implement:
        - start(): Begin producing frames
        - stop(): Stop producing frames
        - is_alive(): Check if source is running
    """

    def __init__(self, output_queue: queue.Queue, config: PipelineConfig):
        """
        Initialize the data source.

        Args:
            output_queue: Queue to push frame data to
            config: Pipeline configuration
        """
        self.output_queue = output_queue
        self.config = config

    @abstractmethod
    def start(self):
        """Start producing frames."""
        pass

    @abstractmethod
    def stop(self):
        """Stop producing frames."""
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the data source is running."""
        pass

    def join(self, timeout: Optional[float] = None):
        """Wait for the data source to stop."""
        pass


class LiveDataSource(DataSource):
    """
    Live radar data from serial port.

    Wraps SerialReader and FrameParser threads to provide
    real radar data through the DataSource interface.
    """

    def __init__(self, output_queue: queue.Queue, config: PipelineConfig):
        super().__init__(output_queue, config)

        # Internal queue between serial reader and frame parser
        self._raw_queue: Optional[queue.Queue] = None
        self._serial_reader = None
        self._frame_parser = None

    def start(self):
        """Start serial reader and frame parser threads."""
        from .serial_reader import SerialReader
        from .frame_parser import FrameParser

        self._raw_queue = queue.Queue(maxsize=self.config.raw_queue_size)

        self._serial_reader = SerialReader(
            port=self.config.serial_port,
            baudrate=self.config.baudrate,
            output_queue=self._raw_queue,
            config=self.config
        )

        self._frame_parser = FrameParser(
            input_queue=self._raw_queue,
            output_queue=self.output_queue,
            config=self.config
        )

        self._serial_reader.start()
        self._frame_parser.start()
        print("[LiveDataSource] Started")

    def stop(self):
        """Stop serial reader and frame parser threads."""
        if self._serial_reader:
            self._serial_reader.stop()
        if self._frame_parser:
            self._frame_parser.stop()
        print("[LiveDataSource] Stopped")

    def is_alive(self) -> bool:
        """Check if threads are running."""
        serial_alive = self._serial_reader and self._serial_reader.is_alive()
        parser_alive = self._frame_parser and self._frame_parser.is_alive()
        return serial_alive or parser_alive

    def join(self, timeout: Optional[float] = None):
        """Wait for threads to stop."""
        if self._serial_reader and self._serial_reader.is_alive():
            self._serial_reader.join(timeout=timeout)
        if self._frame_parser and self._frame_parser.is_alive():
            self._frame_parser.join(timeout=timeout)

    def get_stats(self) -> dict:
        """Get statistics from underlying components."""
        stats = {}
        if self._serial_reader:
            stats['serial'] = self._serial_reader.get_health_stats()
        if self._frame_parser:
            stats['parser'] = self._frame_parser.get_stats()
        return stats


class FileDataSource(DataSource):
    """
    Playback radar data from a JSONL file.

    File format (one JSON object per line):
        {"frame": 1, "ts": 0.0, "detections": [{"x": 0.1, "y": 2.3, "z": 0.5, "snr": 15.2}, ...]}

    Supports:
        - Configurable playback speed
        - Loop at end of file
        - Real-time pacing based on timestamps
    """

    def __init__(self,
                 output_queue: queue.Queue,
                 config: PipelineConfig,
                 file_path: str,
                 speed: float = 1.0,
                 loop: bool = True):
        """
        Initialize file data source.

        Args:
            output_queue: Queue to push frame data to
            config: Pipeline configuration
            file_path: Path to JSONL file
            speed: Playback speed multiplier (1.0 = real-time)
            loop: Whether to loop at end of file
        """
        super().__init__(output_queue, config)
        self.file_path = Path(file_path)
        self.speed = speed
        self.loop = loop

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frames_played = 0
        self._loop_count = 0

    def start(self):
        """Start playback thread."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Recording file not found: {self.file_path}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True, name="FileDataSource")
        self._thread.start()
        print(f"[FileDataSource] Playing {self.file_path} at {self.speed}x speed")

    def stop(self):
        """Stop playback thread."""
        self._stop_event.set()
        print(f"[FileDataSource] Stopped (played {self._frames_played} frames, {self._loop_count} loops)")

    def is_alive(self) -> bool:
        """Check if playback thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None):
        """Wait for playback thread to stop."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _playback_loop(self):
        """Main playback loop - reads file and pushes frames."""
        while not self._stop_event.is_set():
            try:
                self._play_file()
                self._loop_count += 1

                if not self.loop:
                    break

                print(f"[FileDataSource] Looping (loop {self._loop_count})")

            except Exception as e:
                print(f"[FileDataSource] Error during playback: {e}")
                break

    def _play_file(self):
        """Play through the file once."""
        playback_start = time.time()
        first_frame_ts = None

        with open(self.file_path, 'r') as f:
            for line in f:
                if self._stop_event.is_set():
                    return

                line = line.strip()
                if not line:
                    continue

                try:
                    frame_data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[FileDataSource] JSON parse error: {e}")
                    continue

                frame_num = frame_data.get('frame', 0)
                timestamp = frame_data.get('ts', 0.0)
                raw_detections = frame_data.get('detections', [])

                # Convert to Detection objects
                detections = []
                for d in raw_detections:
                    det = Detection(
                        x=d.get('x', 0.0),
                        y=d.get('y', 0.0),
                        z=d.get('z', 0.0),
                        snr=d.get('snr', 0.0),
                        timestamp=timestamp
                    )
                    detections.append(det)

                # Timing: wait until appropriate playback time
                if first_frame_ts is None:
                    first_frame_ts = timestamp
                else:
                    # Calculate how long to wait
                    elapsed_in_file = timestamp - first_frame_ts
                    elapsed_real = time.time() - playback_start
                    target_real = elapsed_in_file / self.speed

                    wait_time = target_real - elapsed_real
                    if wait_time > 0:
                        # Sleep in small increments to check stop event
                        while wait_time > 0 and not self._stop_event.is_set():
                            sleep_time = min(wait_time, 0.01)
                            time.sleep(sleep_time)
                            wait_time -= sleep_time

                if self._stop_event.is_set():
                    return

                # Push frame to queue
                try:
                    self.output_queue.put_nowait((frame_num, timestamp, detections))
                    self._frames_played += 1
                except queue.Full:
                    # Drop frame if queue is full
                    pass

    def get_stats(self) -> dict:
        """Get playback statistics."""
        return {
            'frames_played': self._frames_played,
            'loop_count': self._loop_count,
            'speed': self.speed,
        }


class SyntheticDataSource(DataSource):
    """
    Generate synthetic radar detections for testing.

    Provides various test scenarios without requiring real hardware.
    See scenarios.py for available patterns.
    """

    def __init__(self,
                 output_queue: queue.Queue,
                 config: PipelineConfig,
                 scenario: str = "forward",
                 frame_rate: float = 20.0):
        """
        Initialize synthetic data source.

        Args:
            output_queue: Queue to push frame data to
            config: Pipeline configuration
            scenario: Name of scenario to run (see list_scenarios())
            frame_rate: Frames per second to generate
        """
        super().__init__(output_queue, config)
        self.scenario = scenario
        self.frame_rate = frame_rate

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_count = 0

    @staticmethod
    def list_scenarios() -> List[str]:
        """List available scenarios."""
        from .scenarios import SCENARIOS
        return list(SCENARIOS.keys())

    def start(self):
        """Start generation thread."""
        from .scenarios import SCENARIOS

        if self.scenario not in SCENARIOS:
            available = ', '.join(SCENARIOS.keys())
            raise ValueError(f"Unknown scenario '{self.scenario}'. Available: {available}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._generation_loop, daemon=True, name="SyntheticDataSource")
        self._thread.start()
        print(f"[SyntheticDataSource] Running scenario '{self.scenario}' at {self.frame_rate} Hz")

    def stop(self):
        """Stop generation thread."""
        self._stop_event.set()
        print(f"[SyntheticDataSource] Stopped (generated {self._frame_count} frames)")

    def is_alive(self) -> bool:
        """Check if generation thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None):
        """Wait for generation thread to stop."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _generation_loop(self):
        """Main generation loop."""
        from .scenarios import SCENARIOS

        generator = SCENARIOS[self.scenario]()
        frame_period = 1.0 / self.frame_rate
        start_time = time.time()

        while not self._stop_event.is_set():
            t = time.time() - start_time
            self._frame_count += 1

            # Generate detections for this frame
            detections = generator.generate(t)

            # Push to queue
            try:
                self.output_queue.put_nowait((self._frame_count, t, detections))
            except queue.Full:
                pass

            # Sleep until next frame
            next_frame_time = start_time + (self._frame_count * frame_period)
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            'scenario': self.scenario,
            'frames_generated': self._frame_count,
            'frame_rate': self.frame_rate,
        }
