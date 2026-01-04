#!/usr/bin/env python3
"""
IWR6843ISK Radar Visualizer - PyQtGraph Version
Fast, smooth real-time visualization with noise filtering
"""

import serial
import struct
import time
import numpy as np
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# Serial config
SERIAL_PORT = '/dev/tty.usbserial-00ED1D3E1'
BAUDRATE = 921600

# Magic word to identify frame start
MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# ============================================================================
# FILTERING PARAMETERS - Tune these to reduce noise
# ============================================================================
MIN_RANGE = 0.25          # Ignore detections closer than 25cm
MAX_RANGE = 8.0           # Max detection range
FOV_ANGLE = 60            # Half of total FOV (120° total = ±60° from center)
MIN_SNR = 0.0             # Minimum SNR in dB (higher = stricter)
MIN_CONFIRMATIONS = 5    # Frames a detection must persist before showing
MAX_JUMP_DISTANCE = 0.4   # Max distance an object can "jump" between frames (m)
TRACK_TIMEOUT = 8         # Frames before a track is removed (at 10fps = 0.8 sec)
SMOOTHING_FACTOR = 0.1    # Position smoothing (0=no smoothing, 1=max smoothing)


class TrackedObject:
    """Represents a tracked radar detection with filtering"""
    
    def __init__(self, x, y, z, snr, frame_num):
        self.x = x
        self.y = y
        self.z = z
        self.snr = snr
        self.last_seen = frame_num
        self.confirmations = 1
        self.id = id(self)
        
        # Smoothed position
        self.smooth_x = x
        self.smooth_y = y
        
    @property
    def range(self):
        return np.sqrt(self.smooth_x**2 + self.smooth_y**2 + self.z**2)
    
    @property
    def azimuth_deg(self):
        return np.rad2deg(np.arctan2(self.smooth_x, self.smooth_y))
    
    def update(self, x, y, z, snr, frame_num):
        """Update track with new detection, applying smoothing"""
        # Exponential moving average for smooth motion
        alpha = 1.0 - SMOOTHING_FACTOR
        self.smooth_x = alpha * x + SMOOTHING_FACTOR * self.smooth_x
        self.smooth_y = alpha * y + SMOOTHING_FACTOR * self.smooth_y
        
        self.x = x
        self.y = y
        self.z = z
        self.snr = snr
        self.last_seen = frame_num
        self.confirmations += 1
    
    def distance_to(self, x, y):
        """Distance from this track to a point"""
        return np.sqrt((self.smooth_x - x)**2 + (self.smooth_y - y)**2)
    
    def is_confirmed(self):
        """Has this track been seen enough times to display?"""
        return self.confirmations >= MIN_CONFIRMATIONS
    
    def is_stale(self, current_frame):
        """Has this track timed out?"""
        return (current_frame - self.last_seen) > TRACK_TIMEOUT


class RadarProcessor:
    """Handles radar data parsing and object tracking"""
    
    def __init__(self, port, baudrate):
        self.serial = serial.Serial(port, baudrate, timeout=0.01)
        self.buffer = bytes()
        self.frame_count = 0
        self.tracks = []
        
    def parse_frame(self):
        """Read and parse a radar frame, return tracked objects"""
        # Read available data
        if self.serial.in_waiting > 0:
            self.buffer += self.serial.read(self.serial.in_waiting)
        
        # Find magic word
        magic_idx = self.buffer.find(MAGIC_WORD)
        if magic_idx < 0:
            return None
            
        # Parse header
        if len(self.buffer) < magic_idx + 40:
            return None
            
        header = struct.unpack('<10I', self.buffer[magic_idx:magic_idx+40])
        packet_len = header[3]
        num_detected = header[7]
        num_tlvs = header[8]
        
        # Wait for full packet
        if len(self.buffer) < magic_idx + packet_len:
            return None
        
        # Extract frame
        frame = self.buffer[magic_idx:magic_idx + packet_len]
        self.buffer = self.buffer[magic_idx + packet_len:]
        self.frame_count += 1
        
        # Parse TLVs
        detections = []
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
                        detections.append((x, y, z))
                        
            elif tlv_type == 7:  # Side info (SNR)
                payload = frame[offset+8:offset+tlv_len]
                for i in range(num_detected):
                    obj_offset = i * 4
                    if obj_offset + 4 <= len(payload):
                        snr, _ = struct.unpack('<2H', payload[obj_offset:obj_offset+4])
                        snr_values.append(snr / 10.0)
            
            offset += tlv_len if tlv_len >= 8 else 8
        
        # Merge SNR with detections and filter
        raw_objects = []
        for i, (x, y, z) in enumerate(detections):
            snr = snr_values[i] if i < len(snr_values) else 0
            range_m = np.sqrt(x**2 + y**2 + z**2)
            
            # Calculate azimuth angle (degrees from center)
            azimuth_deg = np.rad2deg(np.arctan2(x, y)) if y > 0 else 90
            
            # Apply filters
            if range_m < MIN_RANGE or range_m > MAX_RANGE:
                continue
            if abs(azimuth_deg) > FOV_ANGLE:  # Outside 120° FOV
                continue
            if snr < MIN_SNR:
                continue
                
            raw_objects.append((x, y, z, snr))
        
        # Update tracking
        self._update_tracks(raw_objects)
        
        # Return only confirmed tracks
        return [t for t in self.tracks if t.is_confirmed()]
    
    def _update_tracks(self, raw_objects):
        """Match detections to existing tracks"""
        matched_tracks = set()
        matched_detections = set()
        
        # Try to match each detection to an existing track
        for i, (x, y, z, snr) in enumerate(raw_objects):
            best_track = None
            best_dist = MAX_JUMP_DISTANCE
            
            for j, track in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                    
                dist = track.distance_to(x, y)
                if dist < best_dist:
                    best_dist = dist
                    best_track = j
            
            if best_track is not None:
                self.tracks[best_track].update(x, y, z, snr, self.frame_count)
                matched_tracks.add(best_track)
                matched_detections.add(i)
        
        # Create new tracks for unmatched detections
        for i, (x, y, z, snr) in enumerate(raw_objects):
            if i not in matched_detections:
                self.tracks.append(TrackedObject(x, y, z, snr, self.frame_count))
        
        # Remove stale tracks
        self.tracks = [t for t in self.tracks if not t.is_stale(self.frame_count)]
    
    def close(self):
        self.serial.close()


class RadarVisualizerQt(QtWidgets.QMainWindow):
    """PyQtGraph-based radar visualizer"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IWR6843ISK Radar Visualizer")
        self.setGeometry(100, 100, 1000, 800)
        
        # Dark theme
        pg.setConfigOptions(antialias=True, background='#1a1a1a', foreground='w')
        
        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Create plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setXRange(-4, 4)
        self.plot_widget.setYRange(-0.5, 6)
        self.plot_widget.setLabel('bottom', 'X Distance', units='m')
        self.plot_widget.setLabel('left', 'Y Distance', units='m')
        self.plot_widget.setTitle("Radar Detection", color='cyan', size='16pt')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)
        
        # Draw radar position
        self.plot_widget.plot([0], [0], pen=None, symbol='t', symbolSize=20, 
                              symbolBrush='cyan', symbolPen='w')
        
        # Draw FOV lines
        fov = 60
        max_r = 6
        self.plot_widget.plot([0, -max_r*np.sin(np.deg2rad(fov))], 
                              [0, max_r*np.cos(np.deg2rad(fov))],
                              pen=pg.mkPen('c', width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.plot([0, max_r*np.sin(np.deg2rad(fov))], 
                              [0, max_r*np.cos(np.deg2rad(fov))],
                              pen=pg.mkPen('c', width=1, style=QtCore.Qt.PenStyle.DashLine))
        
        # Draw range rings
        for r in [1, 2, 3, 4, 5]:
            theta = np.linspace(-np.pi/2, np.pi/2, 50)
            self.plot_widget.plot(r * np.sin(theta), r * np.cos(theta),
                                  pen=pg.mkPen('#333', width=1))
        
        # Scatter plot for detections
        self.scatter = pg.ScatterPlotItem(size=20, pen=pg.mkPen('w', width=2))
        self.plot_widget.addItem(self.scatter)
        
        # Text labels for detections
        self.labels = []
        
        # Info panel
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: white;
                padding: 10px;
                font-family: monospace;
                font-size: 12px;
                border: 1px solid #444;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.info_label)
        
        # Initialize radar
        try:
            self.radar = RadarProcessor(SERIAL_PORT, BAUDRATE)
        except Exception as e:
            self.info_label.setText(f"Error connecting to radar: {e}")
            self.radar = None
            return
        
        # FPS tracking
        self.fps_times = deque(maxlen=30)
        self.last_time = time.time()
        
        # Update timer (50ms = 20 FPS display update)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(50)
        
    def update_display(self):
        if self.radar is None:
            return
            
        # Parse frame
        tracks = self.radar.parse_frame()
        if tracks is None:
            return
        
        # Calculate FPS
        now = time.time()
        self.fps_times.append(now - self.last_time)
        self.last_time = now
        fps = 1.0 / np.mean(self.fps_times) if self.fps_times else 0
        
        # Clear old labels
        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []
        
        if tracks:
            # Prepare scatter data
            spots = []
            for t in tracks:
                # Color by range
                if t.range < 1.0:
                    color = '#ff4444'  # Red - close
                elif t.range < 3.0:
                    color = '#ffff44'  # Yellow - medium
                else:
                    color = '#44ff44'  # Green - far
                
                spots.append({
                    'pos': (t.smooth_x, t.smooth_y),
                    'brush': pg.mkBrush(color),
                    'size': 18,
                })
                
                # Add label
                label = pg.TextItem(f"{t.range:.2f}m", color='w', anchor=(0, 1))
                label.setPos(t.smooth_x + 0.1, t.smooth_y + 0.1)
                self.plot_widget.addItem(label)
                self.labels.append(label)
            
            self.scatter.setData(spots)
            
            # Update info - fixed 5 slots to prevent jumping
            info = f"Frame: {self.radar.frame_count}  |  FPS: {fps:.1f}  |  Tracked: {len(tracks)}\n"
            info += f"{'─' * 50}\n"
            for i in range(5):
                if i < len(tracks):
                    t = tracks[i]
                    info += f"  [{i+1}] {t.range:.2f}m @ {t.azimuth_deg:+.0f}°  (x:{t.smooth_x:+.2f}, y:{t.smooth_y:.2f})  SNR:{t.snr:.0f}dB\n"
                else:
                    info += f"  [{i+1}] ---\n"
            if len(tracks) > 5:
                info += f"  ... and {len(tracks) - 5} more\n"
            self.info_label.setText(info)
        else:
            self.scatter.setData([])
            info = f"Frame: {self.radar.frame_count}  |  FPS: {fps:.1f}  |  Tracked: 0\n"
            info += f"{'─' * 50}\n"
            for i in range(5):
                info += f"  [{i+1}] ---\n"
            self.info_label.setText(info)
    
    def closeEvent(self, event):
        if self.radar:
            self.radar.close()
        event.accept()


def main():
    print("=" * 60)
    print("IWR6843ISK Radar Visualizer (PyQtGraph)")
    print("=" * 60)
    print(f"Range: {MIN_RANGE}m - {MAX_RANGE}m | FOV: ±{FOV_ANGLE}° ({FOV_ANGLE*2}° total)")
    print(f"Confirmations: {MIN_CONFIRMATIONS} frames | SNR: >{MIN_SNR}dB")
    print("=" * 60)
    
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    
    # Dark palette
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, pg.QtGui.QColor('#1a1a1a'))
    palette.setColor(palette.ColorRole.WindowText, pg.QtGui.QColor('white'))
    app.setPalette(palette)
    
    window = RadarVisualizerQt()
    window.show()
    
    app.exec()
    print("\nRadar stopped.")


if __name__ == '__main__':
    main()

