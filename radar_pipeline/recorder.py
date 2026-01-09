"""
Frame Recorder - Record radar detections to JSONL files.

Records frames from any DataSource to a JSONL file for later playback.
Each line is a JSON object with frame number, timestamp, and detections.
"""

import json
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .data_types import Detection
from .data_source import DataSource, FrameData


class FrameRecorder:
    """
    Records frames to a JSONL file.

    Can be used in two modes:
    1. Wrap a DataSource to record while passing through
    2. Standalone, fed frames directly via record_frame()
    """

    def __init__(self, file_path: Optional[str] = None, auto_name: bool = True):
        """
        Initialize the recorder.

        Args:
            file_path: Path to output file. If None and auto_name=True,
                      generates a timestamped filename.
            auto_name: If True and file_path is None, auto-generate filename
        """
        if file_path:
            self.file_path = Path(file_path)
        elif auto_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_path = Path(f"recording_{timestamp}.jsonl")
        else:
            raise ValueError("Must provide file_path or set auto_name=True")

        self._file = None
        self._frames_recorded = 0
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start(self):
        """Open the output file for writing."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.file_path, 'w')
        self._start_time = time.time()
        self._frames_recorded = 0
        print(f"[Recorder] Recording to {self.file_path}")

    def stop(self):
        """Close the output file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

        duration = time.time() - self._start_time if self._start_time else 0
        print(f"[Recorder] Stopped: {self._frames_recorded} frames in {duration:.1f}s")
        print(f"[Recorder] Saved to {self.file_path}")

    def record_frame(self, frame_num: int, timestamp: float, detections: List[Detection]):
        """
        Record a single frame.

        Args:
            frame_num: Frame number
            timestamp: Frame timestamp
            detections: List of Detection objects
        """
        if not self._file:
            return

        # Convert detections to JSON-serializable format
        det_list = []
        for d in detections:
            det_list.append({
                'x': round(d.x, 4),
                'y': round(d.y, 4),
                'z': round(d.z, 4),
                'snr': round(d.snr, 2),
            })

        frame_data = {
            'frame': frame_num,
            'ts': round(timestamp, 4),
            'detections': det_list,
        }

        with self._lock:
            self._file.write(json.dumps(frame_data) + '\n')
            self._frames_recorded += 1

            # Flush periodically
            if self._frames_recorded % 100 == 0:
                self._file.flush()

    @property
    def frames_recorded(self) -> int:
        """Number of frames recorded."""
        return self._frames_recorded


class RecordingDataSource(DataSource):
    """
    Wrapper that records frames while passing them through.

    Wraps another DataSource, recording all frames to a file
    while forwarding them to the output queue.
    """

    def __init__(self,
                 source: DataSource,
                 output_queue: queue.Queue,
                 file_path: Optional[str] = None):
        """
        Initialize recording data source.

        Args:
            source: Underlying data source to wrap
            output_queue: Queue to forward frames to
            file_path: Path to recording file (auto-generated if None)
        """
        # Don't call super().__init__ since we're wrapping
        self._source = source
        self.output_queue = output_queue
        self.config = source.config

        self._recorder = FrameRecorder(file_path)
        self._intercept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Internal queue to intercept frames
        self._intercept_queue = queue.Queue(maxsize=100)

        # Rewire the source to output to our intercept queue
        self._source.output_queue = self._intercept_queue

    def start(self):
        """Start the underlying source and recording."""
        self._recorder.start()
        self._stop_event.clear()

        # Start intercept thread
        self._intercept_thread = threading.Thread(
            target=self._intercept_loop,
            daemon=True,
            name="RecordingIntercept"
        )
        self._intercept_thread.start()

        # Start underlying source
        self._source.start()

    def stop(self):
        """Stop recording and the underlying source."""
        self._source.stop()
        self._stop_event.set()

        if self._intercept_thread:
            self._intercept_thread.join(timeout=1.0)

        self._recorder.stop()

    def is_alive(self) -> bool:
        """Check if source is running."""
        return self._source.is_alive()

    def join(self, timeout: Optional[float] = None):
        """Wait for source to stop."""
        self._source.join(timeout)

    def _intercept_loop(self):
        """Intercept frames, record them, and forward to output."""
        while not self._stop_event.is_set():
            try:
                frame = self._intercept_queue.get(timeout=0.1)
                frame_num, timestamp, detections = frame

                # Record the frame
                self._recorder.record_frame(frame_num, timestamp, detections)

                # Forward to output queue
                try:
                    self.output_queue.put_nowait(frame)
                except queue.Full:
                    pass

            except queue.Empty:
                continue

    def get_stats(self) -> dict:
        """Get recording statistics."""
        stats = {}
        if hasattr(self._source, 'get_stats'):
            stats['source'] = self._source.get_stats()
        stats['recording'] = {
            'frames_recorded': self._recorder.frames_recorded,
            'file_path': str(self._recorder.file_path),
        }
        return stats
