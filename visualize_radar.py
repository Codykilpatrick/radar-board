import serial
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Magic word to identify frame start
MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# Filtering thresholds
MIN_RANGE = 0.1    # Minimum 10cm to filter noise
MAX_RANGE = 10.0   # Max 10 meters
MIN_SNR = 0.0      # Minimum SNR (if available)

# Tracking parameters
PERSISTENCE_FRAMES = 2
MAX_DISTANCE_MATCH = 0.8

class TrackedObject:
    def __init__(self, obj, frame_num):
        self.x = obj['x']
        self.y = obj['y']
        self.range = obj['range']
        self.azimuth_deg = obj['azimuth_deg']
        self.doppler = obj['doppler']
        self.snr = obj['snr']
        self.last_seen = frame_num
        self.age = 1
        
        self.x_history = deque([obj['x']], maxlen=3)
        self.y_history = deque([obj['y']], maxlen=3)
    
    def update(self, obj, frame_num):
        self.x_history.append(obj['x'])
        self.y_history.append(obj['y'])
        self.x = np.mean(self.x_history)
        self.y = np.mean(self.y_history)
        self.range = obj['range']
        self.azimuth_deg = obj['azimuth_deg']
        self.doppler = obj['doppler']
        self.snr = obj['snr']
        self.last_seen = frame_num
        self.age += 1
    
    def is_stale(self, current_frame):
        return (current_frame - self.last_seen) > PERSISTENCE_FRAMES

class RadarVisualizer:
    def __init__(self, port='/dev/tty.usbserial-00ED1D3E1', baudrate=921600):
        self.data_port = serial.Serial(port, baudrate, timeout=0.1)
        self.buffer = bytes()
        
        self.tracked_objects = []
        self.frame_count = 0
        
        self.fps_times = deque(maxlen=30)
        self.last_time = time.time()
        
        # Setup plot
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.ax.set_xlim(-5, 5)      # ±5m lateral
        self.ax.set_ylim(-1, MAX_RANGE)  # -1 to 10m forward
        self.ax.set_xlabel('X Distance (meters)', fontsize=13, color='white')
        self.ax.set_ylabel('Y Distance (meters)', fontsize=13, color='white')
        self.ax.set_title('IWR6843ISK mmWave Radar Visualization', fontsize=16, color='cyan', pad=20)
        self.ax.grid(True, alpha=0.2, color='gray')
        self.ax.set_facecolor('#1a1a1a')
        
        self.ax.axhline(y=0, color='red', linewidth=2, alpha=0.7, label='Range=0')
        self.ax.axvline(x=0, color='cyan', linewidth=1, alpha=0.5)
        
        self.ax.plot(0, 0, 'c^', markersize=20, label='Radar', markeredgecolor='white', markeredgewidth=2)
        
        # FOV visualization
        fov_angle = 60
        self.ax.plot([0, -MAX_RANGE*np.sin(np.deg2rad(fov_angle))], 
                     [0, MAX_RANGE*np.cos(np.deg2rad(fov_angle))], 
                     'c--', alpha=0.3, linewidth=2)
        self.ax.plot([0, MAX_RANGE*np.sin(np.deg2rad(fov_angle))], 
                     [0, MAX_RANGE*np.cos(np.deg2rad(fov_angle))], 
                     'c--', alpha=0.3, linewidth=2)
        
        self.scatter = self.ax.scatter([], [], s=200, alpha=0.8, 
                                      edgecolors='white', linewidth=2, marker='o')
        
        self.annotations = []
        
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                     verticalalignment='top', fontsize=10, color='white',
                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='cyan'))
        
        self.ax.legend(loc='upper right', fontsize=11, facecolor='black', edgecolor='cyan')
        
    def parse_header(self, data):
        if len(data) < 40:
            return None
        
        header = struct.unpack('<10I', data[:40])
        return {
            'totalPacketLen': header[3],
            'frameNumber': header[5],
            'numDetectedObj': header[7],
            'numTLVs': header[8],
        }
    
    def parse_tlvs(self, frame_data, num_objects, num_tlvs):
        """Parse TLVs from the IWR6843ISK out-of-box demo format"""
        objects = []
        snr_data = []
        
        offset = 40  # Skip the 40-byte frame header
        
        for _ in range(num_tlvs):
            if offset + 8 > len(frame_data):
                break
            
            tlv_type, tlv_len = struct.unpack('<2I', frame_data[offset:offset+8])
            
            if tlv_type == 1:  # Detected Points (X, Y, Z as floats - 12 bytes per object)
                payload = frame_data[offset+8:offset+tlv_len]
                for i in range(num_objects):
                    obj_offset = i * 12
                    if obj_offset + 12 <= len(payload):
                        x, y, z = struct.unpack('<3f', payload[obj_offset:obj_offset+12])
                        range_m = np.sqrt(x**2 + y**2 + z**2)
                        azimuth_rad = np.arctan2(x, y) if y != 0 else 0
                        
                        if MIN_RANGE <= range_m <= MAX_RANGE:
                            objects.append({
                                'range': range_m,
                                'azimuth_deg': np.rad2deg(azimuth_rad),
                                'doppler': 0,  # Will be updated if available
                                'snr': 0,  # Will be updated if available
                                'x': x,
                                'y': y,
                                'z': z
                            })
            
            elif tlv_type == 7:  # Side Info for Detected Points (SNR, noise)
                payload = frame_data[offset+8:offset+tlv_len]
                for i in range(num_objects):
                    obj_offset = i * 4
                    if obj_offset + 4 <= len(payload):
                        snr, noise = struct.unpack('<2H', payload[obj_offset:obj_offset+4])
                        snr_data.append(snr / 10.0)  # Convert to dB
            
            offset += tlv_len if tlv_len >= 8 else 8
        
        # Merge SNR data with objects
        for i, obj in enumerate(objects):
            if i < len(snr_data):
                obj['snr'] = snr_data[i]
        
        return objects
    
    def update_tracking(self, new_objects):
        matched_tracks = set()
        matched_objects = set()
        
        for i, track in enumerate(self.tracked_objects):
            best_match = None
            best_dist = MAX_DISTANCE_MATCH
            
            for j, obj in enumerate(new_objects):
                if j in matched_objects:
                    continue
                    
                dist = np.sqrt((track.x - obj['x'])**2 + (track.y - obj['y'])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_match = j
            
            if best_match is not None:
                track.update(new_objects[best_match], self.frame_count)
                matched_tracks.add(i)
                matched_objects.add(best_match)
        
        for j, obj in enumerate(new_objects):
            if j not in matched_objects:
                self.tracked_objects.append(TrackedObject(obj, self.frame_count))
        
        self.tracked_objects = [t for t in self.tracked_objects 
                               if not t.is_stale(self.frame_count)]
    
    def read_frame(self):
        if self.data_port.in_waiting > 0:
            self.buffer += self.data_port.read(self.data_port.in_waiting)
        
        magic_idx = self.buffer.find(MAGIC_WORD)
        if magic_idx >= 0:
            header_data = self.buffer[magic_idx:magic_idx+40]
            header = self.parse_header(header_data)
            
            if header and header['totalPacketLen'] < 10000:
                packet_len = header['totalPacketLen']
                
                timeout = time.time() + 0.5
                while len(self.buffer) < magic_idx + packet_len and time.time() < timeout:
                    if self.data_port.in_waiting > 0:
                        self.buffer += self.data_port.read(self.data_port.in_waiting)
                    time.sleep(0.001)
                
                if len(self.buffer) >= magic_idx + packet_len:
                    frame = self.buffer[magic_idx:magic_idx + packet_len]
                    self.frame_count += 1
                    
                    if header['numDetectedObj'] > 0:
                        new_objects = self.parse_tlvs(frame, header['numDetectedObj'], header['numTLVs'])
                        self.update_tracking(new_objects)
                    
                    self.buffer = self.buffer[magic_idx + packet_len:]
                    return True
        
        return False
    
    def update_plot(self, frame):
        if self.read_frame():
            current_time = time.time()
            self.fps_times.append(current_time - self.last_time)
            self.last_time = current_time
            fps = 1.0 / np.mean(self.fps_times) if self.fps_times else 0
            
            for ann in self.annotations:
                ann.remove()
            self.annotations = []
            
            if self.tracked_objects:
                xs = [t.x for t in self.tracked_objects]
                ys = [t.y for t in self.tracked_objects]
                ranges = [t.range for t in self.tracked_objects]
                
                # Color by range: 0-1m=red (close), 1-3m=yellow (medium), >3m=green (far)
                colors = []
                for r in ranges:
                    if r < 1.0:
                        colors.append('red')
                    elif r < 3.0:
                        colors.append('yellow')
                    else:
                        colors.append('lime')
                
                sizes = [150 for _ in self.tracked_objects]
                
                self.scatter.set_offsets(np.c_[xs, ys])
                self.scatter.set_sizes(sizes)
                self.scatter.set_color(colors)
                
                # Label all objects
                for t in self.tracked_objects:
                    label = f"{t.range:.2f}m"
                    ann = self.ax.annotate(label, (t.x, t.y), 
                                          xytext=(5, 5), textcoords='offset points',
                                          fontsize=8, color='white',
                                          bbox=dict(boxstyle='round,pad=0.2', 
                                                   facecolor='black', alpha=0.6))
                    self.annotations.append(ann)
                
                # Count by range category
                close = sum(1 for t in self.tracked_objects if t.range < 1.0)
                mid = sum(1 for t in self.tracked_objects if 1.0 <= t.range < 3.0)
                far = sum(1 for t in self.tracked_objects if t.range >= 3.0)
                
                info = f'Frame: {self.frame_count} | FPS: {fps:.1f}\n'
                info += f'Tracked Objects: {len(self.tracked_objects)}\n\n'
                info += f'[RED] Close (<1m): {close}\n'
                info += f'[YLW] Medium (1-3m): {mid}\n'
                info += f'[GRN] Far (>3m): {far}\n\n'
                
                # Show details for tracked objects
                info += 'Detections:\n'
                for i, t in enumerate(self.tracked_objects[:5]):
                    info += f'{i+1}. {t.range:.2f}m @ {t.azimuth_deg:.0f}° (x:{t.x:.2f}, y:{t.y:.2f})\n'
                
                self.info_text.set_text(info)
            else:
                self.scatter.set_offsets(np.empty((0, 2)))
                self.info_text.set_text(f'Frame: {self.frame_count} | FPS: {fps:.1f}\n\nNo objects')
        
        return self.scatter, self.info_text, *self.annotations
    
    def run(self):
        print("=" * 60)
        print("IWR6843ISK mmWave Radar Visualization")
        print("=" * 60)
        print("Format: X, Y, Z Cartesian coordinates (meters)")
        print("Colors: Red=<1m, Yellow=1-3m, Green=>3m")
        print("\nStarting radar visualization...\n")
        
        ani = FuncAnimation(self.fig, self.update_plot, interval=100, 
                          blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        
        self.data_port.close()
        print("\nRadar stopped.")

if __name__ == '__main__':
    try:
        viz = RadarVisualizer()
        viz.run()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
