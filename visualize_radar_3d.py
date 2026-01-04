#!/usr/bin/env python3
"""
IWR6843ISK Radar Visualizer - 3D Version
Uses PyQtGraph OpenGL for 3D visualization
"""

import serial
import struct
import time
import numpy as np
from collections import deque

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

# Serial config
SERIAL_PORT = '/dev/tty.usbserial-00ED1D3E1'
BAUDRATE = 921600

# Magic word to identify frame start
MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# ============================================================================
# FILTERING PARAMETERS
# ============================================================================
MIN_RANGE = 0.25          # Ignore detections closer than 25cm
MAX_RANGE = 8.0           # Max detection range
FOV_ANGLE = 60            # Half of total FOV (120° total = ±60° from center)
MIN_SNR = 0.0             # Minimum SNR in dB
MIN_CONFIRMATIONS = 5     # Frames a detection must persist before showing
MAX_JUMP_DISTANCE = 0.4   # Max distance an object can "jump" between frames (m)
TRACK_TIMEOUT = 8         # Frames before a track is removed
SMOOTHING_FACTOR = 0.1    # Position smoothing


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
        self.smooth_z = z
        
    @property
    def range(self):
        return np.sqrt(self.smooth_x**2 + self.smooth_y**2 + self.smooth_z**2)
    
    @property
    def azimuth_deg(self):
        return np.rad2deg(np.arctan2(self.smooth_x, self.smooth_y))
    
    @property
    def elevation_deg(self):
        xy_dist = np.sqrt(self.smooth_x**2 + self.smooth_y**2)
        return np.rad2deg(np.arctan2(self.smooth_z, xy_dist))
    
    def update(self, x, y, z, snr, frame_num):
        """Update track with new detection, applying smoothing"""
        alpha = 1.0 - SMOOTHING_FACTOR
        self.smooth_x = alpha * x + SMOOTHING_FACTOR * self.smooth_x
        self.smooth_y = alpha * y + SMOOTHING_FACTOR * self.smooth_y
        self.smooth_z = alpha * z + SMOOTHING_FACTOR * self.smooth_z
        
        self.x = x
        self.y = y
        self.z = z
        self.snr = snr
        self.last_seen = frame_num
        self.confirmations += 1
    
    def distance_to(self, x, y, z):
        """3D distance from this track to a point"""
        return np.sqrt((self.smooth_x - x)**2 + (self.smooth_y - y)**2 + (self.smooth_z - z)**2)
    
    def is_confirmed(self):
        return self.confirmations >= MIN_CONFIRMATIONS
    
    def is_stale(self, current_frame):
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
        if self.serial.in_waiting > 0:
            self.buffer += self.serial.read(self.serial.in_waiting)
        
        magic_idx = self.buffer.find(MAGIC_WORD)
        if magic_idx < 0:
            return None
            
        if len(self.buffer) < magic_idx + 40:
            return None
            
        header = struct.unpack('<10I', self.buffer[magic_idx:magic_idx+40])
        packet_len = header[3]
        num_detected = header[7]
        num_tlvs = header[8]
        
        if len(self.buffer) < magic_idx + packet_len:
            return None
        
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
        
        # Filter detections
        raw_objects = []
        for i, (x, y, z) in enumerate(detections):
            snr = snr_values[i] if i < len(snr_values) else 0
            range_m = np.sqrt(x**2 + y**2 + z**2)
            azimuth_deg = np.rad2deg(np.arctan2(x, y)) if y > 0 else 90
            
            if range_m < MIN_RANGE or range_m > MAX_RANGE:
                continue
            if abs(azimuth_deg) > FOV_ANGLE:
                continue
            if snr < MIN_SNR:
                continue
                
            raw_objects.append((x, y, z, snr))
        
        self._update_tracks(raw_objects)
        return [t for t in self.tracks if t.is_confirmed()]
    
    def _update_tracks(self, raw_objects):
        """Match detections to existing tracks"""
        matched_tracks = set()
        matched_detections = set()
        
        for i, (x, y, z, snr) in enumerate(raw_objects):
            best_track = None
            best_dist = MAX_JUMP_DISTANCE
            
            for j, track in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                    
                dist = track.distance_to(x, y, z)
                if dist < best_dist:
                    best_dist = dist
                    best_track = j
            
            if best_track is not None:
                self.tracks[best_track].update(x, y, z, snr, self.frame_count)
                matched_tracks.add(best_track)
                matched_detections.add(i)
        
        for i, (x, y, z, snr) in enumerate(raw_objects):
            if i not in matched_detections:
                self.tracks.append(TrackedObject(x, y, z, snr, self.frame_count))
        
        self.tracks = [t for t in self.tracks if not t.is_stale(self.frame_count)]
    
    def close(self):
        self.serial.close()


class RadarVisualizer3D(QtWidgets.QMainWindow):
    """3D PyQtGraph OpenGL radar visualizer"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IWR6843ISK 3D Radar Visualizer")
        self.setGeometry(100, 100, 1200, 900)
        
        # Central widget with layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # 3D view widget
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=8, elevation=25, azimuth=-90)
        self.view.setBackgroundColor('#1a1a1a')
        layout.addWidget(self.view, stretch=4)
        
        # Add ground grid
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        grid.translate(0, 5, 0)
        self.view.addItem(grid)
        
        # Add axis reference at radar position
        axis = gl.GLAxisItem()
        axis.setSize(0.5, 0.5, 0.5)
        self.view.addItem(axis)
        
        # Add radar marker (cyan pyramid)
        radar_verts = np.array([
            [0, 0, 0],
            [-0.1, 0.15, 0],
            [0.1, 0.15, 0],
            [0, 0.15, 0.1],
        ])
        radar_faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ])
        radar_mesh = gl.GLMeshItem(vertexes=radar_verts, faces=radar_faces,
                                    color=(0, 1, 1, 1), smooth=False)
        self.view.addItem(radar_mesh)
        
        # Draw FOV cone edges
        fov_rad = np.deg2rad(FOV_ANGLE)
        max_r = 6
        fov_lines = np.array([
            [[0, 0, 0], [-max_r * np.sin(fov_rad), max_r * np.cos(fov_rad), 0]],
            [[0, 0, 0], [max_r * np.sin(fov_rad), max_r * np.cos(fov_rad), 0]],
            [[0, 0, 0], [0, max_r * np.cos(fov_rad), max_r * np.sin(fov_rad) * 0.5]],
            [[0, 0, 0], [0, max_r * np.cos(fov_rad), -max_r * np.sin(fov_rad) * 0.5]],
        ])
        for line in fov_lines:
            fov_item = gl.GLLinePlotItem(pos=line, color=(0, 1, 1, 0.3), width=1)
            self.view.addItem(fov_item)
        
        # Add range rings on ground
        for r in [1, 2, 3, 4, 5]:
            theta = np.linspace(-np.pi/2, np.pi/2, 30)
            ring = np.column_stack([
                r * np.sin(theta),
                r * np.cos(theta),
                np.zeros_like(theta)
            ])
            ring_item = gl.GLLinePlotItem(pos=ring, color=(0.3, 0.3, 0.3, 1), width=1)
            self.view.addItem(ring_item)
        
        # Scatter plot for detections
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 0, 0, 0), size=0)
        self.view.addItem(self.scatter)
        
        # Vertical lines from ground to points (for depth perception)
        self.vert_lines = []
        
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
        
        # Update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(50)
        
    def update_display(self):
        if self.radar is None:
            return
            
        tracks = self.radar.parse_frame()
        if tracks is None:
            return
        
        # Calculate FPS
        now = time.time()
        self.fps_times.append(now - self.last_time)
        self.last_time = now
        fps = 1.0 / np.mean(self.fps_times) if self.fps_times else 0
        
        # Clear old vertical lines
        for line in self.vert_lines:
            self.view.removeItem(line)
        self.vert_lines = []
        
        if tracks:
            # Prepare 3D scatter data
            positions = np.array([[t.smooth_x, t.smooth_y, t.smooth_z] for t in tracks])
            
            # Color by range
            colors = []
            for t in tracks:
                if t.range < 1.0:
                    colors.append((1, 0.3, 0.3, 1))  # Red
                elif t.range < 3.0:
                    colors.append((1, 1, 0.3, 1))    # Yellow
                else:
                    colors.append((0.3, 1, 0.3, 1))  # Green
            
            colors = np.array(colors)
            sizes = np.full(len(tracks), 15)
            
            self.scatter.setData(pos=positions, color=colors, size=sizes)
            
            # Add vertical lines for depth perception
            for t in tracks:
                line_pos = np.array([
                    [t.smooth_x, t.smooth_y, 0],
                    [t.smooth_x, t.smooth_y, t.smooth_z]
                ])
                line = gl.GLLinePlotItem(pos=line_pos, color=(0.5, 0.5, 0.5, 0.5), width=1)
                self.view.addItem(line)
                self.vert_lines.append(line)
            
            # Update info - fixed 5 slots
            info = f"Frame: {self.radar.frame_count}  |  FPS: {fps:.1f}  |  Tracked: {len(tracks)}\n"
            info += f"{'─' * 60}\n"
            for i in range(5):
                if i < len(tracks):
                    t = tracks[i]
                    info += f"  [{i+1}] {t.range:.2f}m  Az:{t.azimuth_deg:+.0f}°  El:{t.elevation_deg:+.0f}°  (x:{t.smooth_x:+.2f}, y:{t.smooth_y:.2f}, z:{t.smooth_z:+.2f})\n"
                else:
                    info += f"  [{i+1}] ---\n"
            if len(tracks) > 5:
                info += f"  ... and {len(tracks) - 5} more\n"
            self.info_label.setText(info)
        else:
            self.scatter.setData(pos=np.zeros((1, 3)), color=(1, 0, 0, 0), size=0)
            info = f"Frame: {self.radar.frame_count}  |  FPS: {fps:.1f}  |  Tracked: 0\n"
            info += f"{'─' * 60}\n"
            for i in range(5):
                info += f"  [{i+1}] ---\n"
            self.info_label.setText(info)
    
    def closeEvent(self, event):
        if self.radar:
            self.radar.close()
        event.accept()


def main():
    print("=" * 60)
    print("IWR6843ISK 3D Radar Visualizer")
    print("=" * 60)
    print(f"Range: {MIN_RANGE}m - {MAX_RANGE}m | FOV: ±{FOV_ANGLE}°")
    print("Controls: Left-drag to rotate, Right-drag to zoom, Middle-drag to pan")
    print("=" * 60)
    
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QtGui.QColor('#1a1a1a'))
    palette.setColor(palette.ColorRole.WindowText, QtGui.QColor('white'))
    app.setPalette(palette)
    
    window = RadarVisualizer3D()
    window.show()
    
    app.exec()
    print("\nRadar stopped.")


if __name__ == '__main__':
    main()

