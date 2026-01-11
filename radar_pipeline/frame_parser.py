"""
Frame Parser Thread - Parses raw radar frames into detections.

Responsibilities:
- Consume raw frame bytes from input queue
- Parse TLV structure
- Extract detections with coordinates and SNR
- Apply basic filtering (range gate, FOV, SNR threshold)
- Output detection lists with timestamp
"""

import threading
import queue
import struct
import time
import numpy as np
from typing import List, Optional, TYPE_CHECKING

from .data_types import RawFrame, Detection
from .config import PipelineConfig

if TYPE_CHECKING:
    from .background import BackgroundModel


class FrameParser(threading.Thread):
    """
    Thread that parses raw radar frames into Detection objects.
    
    Consumes RawFrame objects from input queue, parses TLV structure,
    applies filtering, and outputs Detection lists to output queue.
    """
    
    def __init__(self,
                 input_queue: queue.Queue,
                 output_queue: queue.Queue,
                 config: PipelineConfig,
                 background_model: Optional['BackgroundModel'] = None):
        """
        Initialize the frame parser thread.

        Args:
            input_queue: Queue to consume RawFrame objects from
            output_queue: Queue to push List[Detection] to
            config: Pipeline configuration
            background_model: Optional background model for clutter filtering
        """
        super().__init__(daemon=True, name="FrameParser")

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.background_model = background_model

        self._stop_event = threading.Event()

        # Stats
        self._frames_parsed = 0
        self._detections_total = 0
        self._detections_filtered = 0
        self._bg_filtered = 0
        
    def run(self):
        """Main thread loop - consumes raw frames and parses them."""
        print("[FrameParser] Started")
        
        while not self._stop_event.is_set():
            try:
                # Get raw frame with timeout
                raw_frame = self.input_queue.get(timeout=0.1)
                
                # Parse the frame
                detections = self._parse_frame(raw_frame)
                
                # Push detections to output queue
                if detections is not None:
                    try:
                        self.output_queue.put_nowait((raw_frame.frame_number, 
                                                       raw_frame.timestamp, 
                                                       detections))
                    except queue.Full:
                        pass  # Drop if queue is full
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FrameParser] Error parsing frame: {e}")
                continue
        
        print("[FrameParser] Stopping...")
    
    def _parse_frame(self, raw_frame: RawFrame) -> Optional[List[Detection]]:
        """
        Parse a raw frame into a list of detections.
        
        Args:
            raw_frame: Raw frame data
        
        Returns:
            List of Detection objects, or None if parsing fails
        """
        frame = raw_frame.data
        timestamp = raw_frame.timestamp
        
        if len(frame) < 40:
            return None
        
        try:
            # Parse header
            header = struct.unpack('<10I', frame[:40])
            num_detected = header[7]
            num_tlvs = header[8]
            
            self._frames_parsed += 1
            
            # Parse TLVs
            detections_raw = []
            snr_values = []
            offset = 40
            
            for _ in range(num_tlvs):
                if offset + 8 > len(frame):
                    break
                
                tlv_type, tlv_len = struct.unpack('<2I', frame[offset:offset+8])
                
                if tlv_type == 1:  # Detected points (X, Y, Z)
                    payload = frame[offset+8:offset+tlv_len]
                    for i in range(num_detected):
                        obj_offset = i * 12
                        if obj_offset + 12 <= len(payload):
                            x, y, z = struct.unpack('<3f', payload[obj_offset:obj_offset+12])
                            detections_raw.append((x, y, z))
                
                elif tlv_type == 7:  # Side info (SNR)
                    payload = frame[offset+8:offset+tlv_len]
                    for i in range(num_detected):
                        obj_offset = i * 4
                        if obj_offset + 4 <= len(payload):
                            snr, _ = struct.unpack('<2H', payload[obj_offset:obj_offset+4])
                            snr_values.append(snr / 10.0)  # Convert to dB
                
                # Move to next TLV
                offset += tlv_len if tlv_len >= 8 else 8
            
            # Apply filtering and create Detection objects
            detections = self._filter_detections(detections_raw, snr_values, timestamp)
            
            return detections
            
        except struct.error as e:
            print(f"[FrameParser] Struct error: {e}")
            return None
    
    def _filter_detections(self, 
                           raw_detections: List[tuple],
                           snr_values: List[float],
                           timestamp: float) -> List[Detection]:
        """
        Filter raw detections based on configuration.
        
        Args:
            raw_detections: List of (x, y, z) tuples
            snr_values: List of SNR values (may be shorter than detections)
            timestamp: Frame timestamp
        
        Returns:
            List of filtered Detection objects
        """
        detections = []
        
        min_range = self.config.min_range
        max_range = self.config.max_range
        fov_angle = self.config.fov_angle
        min_snr = self.config.min_snr
        
        for i, (x, y, z) in enumerate(raw_detections):
            snr = snr_values[i] if i < len(snr_values) else 0.0
            
            # Calculate range
            range_m = np.sqrt(x**2 + y**2 + z**2)
            
            # Range filter
            if range_m < min_range or range_m > max_range:
                self._detections_filtered += 1
                continue
            
            # FOV filter
            azimuth_deg = np.rad2deg(np.arctan2(x, y)) if y > 0 else 90.0
            if abs(azimuth_deg) > fov_angle:
                self._detections_filtered += 1
                continue
            
            # SNR filter
            if snr < min_snr:
                self._detections_filtered += 1
                continue

            # Background filter
            if self.background_model and self.background_model.is_background(x, y, z):
                self._bg_filtered += 1
                continue

            # Passed all filters
            detection = Detection(
                x=x,
                y=y,
                z=z,
                snr=snr,
                timestamp=timestamp
            )
            detections.append(detection)
            self._detections_total += 1
        
        # Debug output for SNR and detection values
        if self.config.debug_detections and raw_detections:
            snr_list = [snr_values[i] if i < len(snr_values) else 0.0 for i in range(len(raw_detections))]
            if snr_list:
                snr_min, snr_max = min(snr_list), max(snr_list)
                snr_mean = sum(snr_list) / len(snr_list)
            else:
                snr_min = snr_max = snr_mean = 0.0
            
            print(f"[Detect] raw={len(raw_detections)} passed={len(detections)} | "
                  f"SNR: min={snr_min:.1f} max={snr_max:.1f} mean={snr_mean:.1f} dB")
        
        return detections
    
    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()
    
    def get_stats(self) -> dict:
        """Get parsing statistics."""
        return {
            'frames_parsed': self._frames_parsed,
            'detections_total': self._detections_total,
            'detections_filtered': self._detections_filtered,
            'bg_filtered': self._bg_filtered,
        }

