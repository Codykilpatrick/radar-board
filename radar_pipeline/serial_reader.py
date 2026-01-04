"""
Serial Reader Thread - Reads raw bytes from radar serial port.

Responsibilities:
- Read raw bytes from serial port in dedicated thread
- Find frame boundaries using magic word
- Push complete raw frames to queue
- Handle serial errors gracefully
"""

import threading
import queue
import time
import serial
from typing import Optional
from collections import deque

from .data_types import RawFrame
from .config import PipelineConfig


class SerialReader(threading.Thread):
    """
    Thread that reads raw frames from the radar serial port.
    
    Finds frame boundaries using magic word and pushes complete frames
    to the output queue for parsing.
    """
    
    def __init__(self, 
                 port: str,
                 baudrate: int,
                 output_queue: queue.Queue,
                 config: PipelineConfig):
        """
        Initialize the serial reader thread.
        
        Args:
            port: Serial port path
            baudrate: Serial baudrate
            output_queue: Queue to push RawFrame objects
            config: Pipeline configuration
        """
        super().__init__(daemon=True, name="SerialReader")
        
        self.port = port
        self.baudrate = baudrate
        self.output_queue = output_queue
        self.config = config
        
        self._stop_event = threading.Event()
        self._serial: Optional[serial.Serial] = None
        self._buffer = bytes()
        self._frame_count = 0
        
        # Health metrics
        self._bytes_received = 0
        self._bytes_times = deque(maxlen=100)
        self._frames_pushed = 0
        self._dropped_frames = 0
        
    @property
    def bytes_per_second(self) -> float:
        """Estimate current bytes per second throughput."""
        if len(self._bytes_times) < 2:
            return 0.0
        times = list(self._bytes_times)
        elapsed = times[-1][1] - times[0][1]
        if elapsed <= 0:
            return 0.0
        total_bytes = sum(b for b, _ in times)
        return total_bytes / elapsed
    
    @property
    def frames_per_second(self) -> float:
        """Frames pushed to queue per second."""
        # Rough estimate based on frame rate
        return 20.0 if self._frames_pushed > 0 else 0.0
    
    @property
    def buffer_size(self) -> int:
        """Current buffer size in bytes."""
        return len(self._buffer)
    
    @property
    def total_frames(self) -> int:
        """Total frames extracted."""
        return self._frame_count
    
    @property
    def dropped_frames(self) -> int:
        """Number of dropped frames (queue full)."""
        return self._dropped_frames
    
    def run(self):
        """Main thread loop - reads serial data and extracts frames."""
        try:
            self._serial = serial.Serial(
                self.port, 
                self.baudrate, 
                timeout=0.01
            )
            print(f"[SerialReader] Connected to {self.port} at {self.baudrate} baud")
        except serial.SerialException as e:
            print(f"[SerialReader] Failed to open serial port: {e}")
            return
        
        magic_word = self.config.magic_word
        
        while not self._stop_event.is_set():
            try:
                # Read available bytes
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    self._buffer += data
                    self._bytes_received += len(data)
                    self._bytes_times.append((len(data), time.time()))
                
                # Extract complete frames from buffer
                self._extract_frames(magic_word)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except serial.SerialException as e:
                print(f"[SerialReader] Serial error: {e}")
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[SerialReader] Unexpected error: {e}")
                time.sleep(0.1)
        
        print("[SerialReader] Stopping...")
        if self._serial:
            self._serial.close()
    
    def _extract_frames(self, magic_word: bytes):
        """
        Extract complete frames from buffer.
        
        Finds magic word, reads header to get packet length,
        extracts complete frame and pushes to queue.
        """
        while True:
            # Find magic word
            magic_idx = self._buffer.find(magic_word)
            if magic_idx < 0:
                # No magic word found, keep last 7 bytes (partial magic)
                if len(self._buffer) > 7:
                    self._buffer = self._buffer[-7:]
                break
            
            # Discard bytes before magic word
            if magic_idx > 0:
                self._buffer = self._buffer[magic_idx:]
            
            # Need at least 40 bytes for header
            if len(self._buffer) < 40:
                break
            
            # Parse header to get packet length
            # Header format: 10 x 32-bit values
            # [0-1]: magic word parts
            # [2]: version
            # [3]: total packet length
            # [4]: platform
            # [5]: frame number
            # [6]: time (CPU cycles)
            # [7]: number of detected objects
            # [8]: number of TLVs
            # [9]: subframe number
            import struct
            try:
                header = struct.unpack('<10I', self._buffer[:40])
                packet_len = header[3]
                
                # Sanity check on packet length
                if packet_len < 40 or packet_len > 65535:
                    # Invalid packet, skip this magic word
                    self._buffer = self._buffer[8:]
                    continue
                
                # Check if we have complete frame
                if len(self._buffer) < packet_len:
                    break
                
                # Extract frame
                frame_data = self._buffer[:packet_len]
                self._buffer = self._buffer[packet_len:]
                self._frame_count += 1
                
                # Create RawFrame and push to queue
                raw_frame = RawFrame(
                    data=frame_data,
                    timestamp=time.time(),
                    frame_number=self._frame_count
                )
                
                try:
                    self.output_queue.put_nowait(raw_frame)
                    self._frames_pushed += 1
                except queue.Full:
                    self._dropped_frames += 1
                    
            except struct.error:
                # Malformed header, skip
                self._buffer = self._buffer[8:]
    
    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()
    
    def get_health_stats(self) -> dict:
        """Get current health statistics."""
        return {
            'bytes_per_sec': self.bytes_per_second,
            'buffer_size': self.buffer_size,
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'frames_pushed': self._frames_pushed,
        }

