"""
Video Recorder - Synchronized camera recording alongside radar data.

Records video from webcam with timestamps for synchronization with radar JSONL files.
"""

import cv2
import json
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class VideoFrame:
    """A video frame with timestamp."""
    frame: any  # numpy array
    timestamp: float
    frame_number: int


class VideoRecorder:
    """
    Records video from webcam with timestamps for radar synchronization.

    Creates two files:
    - {name}.mp4 - The video file
    - {name}_timestamps.json - Frame timestamps for sync

    Usage:
        recorder = VideoRecorder("recordings/session_001")
        recorder.start()
        # ... run radar ...
        recorder.stop()
    """

    def __init__(self,
                 output_path: str,
                 camera_index: int = 0,
                 fps: float = 20.0,
                 resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize video recorder.

        Args:
            output_path: Base path for output files (without extension)
            camera_index: Camera device index (0 = default webcam)
            fps: Target frames per second
            resolution: Video resolution (width, height)
        """
        self.output_path = Path(output_path)
        self.camera_index = camera_index
        self.target_fps = fps
        self.resolution = resolution

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.video_path = self.output_path.with_suffix('.mp4')
        self.timestamps_path = Path(str(self.output_path) + '_timestamps.json')

        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._timestamps = []
        self._frame_count = 0
        self._start_time = 0.0

    def start(self) -> bool:
        """
        Start video recording.

        Returns:
            True if camera opened successfully, False otherwise
        """
        # Open camera
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            print(f"[Video] Failed to open camera {self.camera_index}")
            return False

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Get actual resolution (may differ from requested)
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.target_fps,
            (actual_width, actual_height)
        )

        if not self._writer.isOpened():
            print(f"[Video] Failed to create video writer")
            self._cap.release()
            return False

        print(f"[Video] Recording to {self.video_path}")
        print(f"[Video] Resolution: {actual_width}x{actual_height} @ {self.target_fps}fps")

        self._timestamps = []
        self._frame_count = 0
        self._start_time = time.time()
        self._running = True

        # Start capture thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def stop(self):
        """Stop video recording and save timestamps."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._writer:
            self._writer.release()
            self._writer = None

        if self._cap:
            self._cap.release()
            self._cap = None

        # Save timestamps
        if self._timestamps:
            metadata = {
                'start_time': self._start_time,
                'frame_count': self._frame_count,
                'target_fps': self.target_fps,
                'resolution': list(self.resolution),
                'timestamps': self._timestamps
            }
            with open(self.timestamps_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            duration = self._timestamps[-1] - self._timestamps[0] if len(self._timestamps) > 1 else 0
            actual_fps = self._frame_count / duration if duration > 0 else 0
            print(f"[Video] Saved {self._frame_count} frames ({duration:.1f}s, {actual_fps:.1f} fps)")
            print(f"[Video] Timestamps saved to {self.timestamps_path}")

    def _capture_loop(self):
        """Background thread for capturing frames."""
        frame_interval = 1.0 / self.target_fps
        next_frame_time = time.time()

        while self._running:
            now = time.time()

            # Capture at target FPS
            if now >= next_frame_time:
                ret, frame = self._cap.read()
                if ret:
                    timestamp = time.time()
                    self._writer.write(frame)
                    self._timestamps.append(timestamp)
                    self._frame_count += 1

                next_frame_time = now + frame_interval
            else:
                # Small sleep to avoid busy-waiting
                time.sleep(0.001)

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._running


class VideoPlayer:
    """
    Plays back recorded video synchronized with radar data.

    Usage:
        player = VideoPlayer("recordings/session_001")
        player.start()

        # In your replay loop:
        frame = player.get_frame_at(radar_timestamp)
        if frame is not None:
            cv2.imshow("Video", frame)
    """

    def __init__(self, video_path: str):
        """
        Initialize video player.

        Args:
            video_path: Base path (without extension) or path to .mp4 file
        """
        video_path = Path(video_path)

        # Handle both "session" and "session.mp4" inputs
        if video_path.suffix == '.mp4':
            self.video_path = video_path
            self.timestamps_path = Path(str(video_path)[:-4] + '_timestamps.json')
        else:
            self.video_path = video_path.with_suffix('.mp4')
            self.timestamps_path = Path(str(video_path) + '_timestamps.json')

        self._cap: Optional[cv2.VideoCapture] = None
        self._timestamps = []
        self._current_frame = 0
        self._start_time = 0.0

    def start(self) -> bool:
        """
        Open video and load timestamps.

        Returns:
            True if successful, False otherwise
        """
        if not self.video_path.exists():
            print(f"[Video] Video file not found: {self.video_path}")
            return False

        if not self.timestamps_path.exists():
            print(f"[Video] Timestamps file not found: {self.timestamps_path}")
            return False

        # Load timestamps
        with open(self.timestamps_path, 'r') as f:
            metadata = json.load(f)

        self._timestamps = metadata['timestamps']
        self._start_time = metadata['start_time']

        # Open video
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            print(f"[Video] Failed to open video: {self.video_path}")
            return False

        print(f"[Video] Loaded {len(self._timestamps)} frames from {self.video_path}")
        return True

    def stop(self):
        """Release video resources."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame_at(self, timestamp: float) -> Optional[any]:
        """
        Get the video frame closest to the given timestamp.

        Args:
            timestamp: Unix timestamp to match

        Returns:
            Video frame (numpy array) or None if not available
        """
        if not self._cap or not self._timestamps:
            return None

        # Find closest timestamp
        best_idx = 0
        best_diff = abs(self._timestamps[0] - timestamp)

        for i, ts in enumerate(self._timestamps):
            diff = abs(ts - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        # Seek if needed
        if best_idx != self._current_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
            self._current_frame = best_idx

        ret, frame = self._cap.read()
        if ret:
            self._current_frame += 1
            return frame

        return None

    def get_frame_by_index(self, index: int) -> Optional[any]:
        """Get frame by index."""
        if not self._cap or index < 0 or index >= len(self._timestamps):
            return None

        if index != self._current_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._current_frame = index

        ret, frame = self._cap.read()
        if ret:
            self._current_frame += 1
            return frame

        return None

    def get_timestamp(self, index: int) -> Optional[float]:
        """Get timestamp for frame index."""
        if 0 <= index < len(self._timestamps):
            return self._timestamps[index]
        return None

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return len(self._timestamps)

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        if len(self._timestamps) > 1:
            return self._timestamps[-1] - self._timestamps[0]
        return 0.0


def check_camera_available(camera_index: int = 0) -> bool:
    """Check if a camera is available."""
    cap = cv2.VideoCapture(camera_index)
    available = cap.isOpened()
    cap.release()
    return available
