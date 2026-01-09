"""
Render Engine - Qt/OpenGL 3D visualization.

Responsibilities:
- Consume WorldState from queue
- Render 3D visualization using PyQtGraph OpenGL
- Display tracks with color coding by classification
- Show velocity vectors for dynamic objects
- Render static object map
- Display track trails (history)
- Show info panel with track details
"""

import queue
import time
import numpy as np
from typing import List, Optional
from collections import deque

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from .data_types import WorldState, Track, StaticObject, ObjectClass
from .config import PipelineConfig
from .utils import color_by_range, color_by_classification

# Optional OpenCV for video playback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RenderEngine(QtWidgets.QMainWindow):
    """
    Qt/OpenGL 3D visualization for radar tracking.
    
    Consumes WorldState from queue and renders:
    - Dynamic tracked objects as colored spheres
    - Velocity vectors as arrows
    - Static objects as gray markers
    - Track trails showing recent history
    - Info panel with track details
    """
    
    def __init__(self,
                 state_queue: queue.Queue,
                 config: PipelineConfig,
                 video_player=None):
        """
        Initialize the render engine.

        Args:
            state_queue: Queue to consume WorldState from
            config: Pipeline configuration
            video_player: Optional VideoPlayer for synchronized video display
        """
        super().__init__()

        self.state_queue = state_queue
        self.config = config
        self.video_player = video_player

        self._last_state: Optional[WorldState] = None
        self._fps_times = deque(maxlen=30)
        self._video_window_created = False
        
        # Dynamic visualization items that get updated
        self._vert_lines = []
        self._velocity_arrows = []
        self._trail_items = []
        self._static_scatter: Optional[gl.GLScatterPlotItem] = None
        
        self._setup_ui()
        self._setup_3d_view()
        self._setup_timer()
    
    def _setup_ui(self):
        """Set up the main window UI."""
        self.setWindowTitle("IWR6843ISK Radar Pipeline - 3D Visualizer")
        self.setGeometry(100, 100, 1400, 1000)
        
        # Central widget with layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 3D view widget
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=10, elevation=30, azimuth=-90)
        self.view.setBackgroundColor('#0a0a12')
        layout.addWidget(self.view, stretch=5)
        
        # Info panel
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #141420;
                color: #e0e0e0;
                padding: 12px;
                font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
                font-size: 11px;
                border: 1px solid #2a2a3a;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.info_label)
    
    def _setup_3d_view(self):
        """Set up the 3D visualization elements."""
        # Ground grid
        grid = gl.GLGridItem()
        grid.setSize(12, 12)
        grid.setSpacing(1, 1)
        grid.translate(0, 6, 0)
        grid.setColor((0.2, 0.2, 0.25, 0.5))
        self.view.addItem(grid)
        
        # Axis reference at radar position
        axis = gl.GLAxisItem()
        axis.setSize(0.5, 0.5, 0.5)
        self.view.addItem(axis)
        
        # Radar marker (cyan pyramid)
        radar_verts = np.array([
            [0, 0, 0],
            [-0.12, 0.18, 0],
            [0.12, 0.18, 0],
            [0, 0.18, 0.12],
        ])
        radar_faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ])
        radar_mesh = gl.GLMeshItem(
            vertexes=radar_verts, 
            faces=radar_faces,
            color=(0, 0.9, 0.9, 1), 
            smooth=False
        )
        self.view.addItem(radar_mesh)
        
        # FOV cone edges
        fov_rad = np.deg2rad(self.config.fov_angle)
        max_r = 7
        fov_lines = np.array([
            [[0, 0, 0], [-max_r * np.sin(fov_rad), max_r * np.cos(fov_rad), 0]],
            [[0, 0, 0], [max_r * np.sin(fov_rad), max_r * np.cos(fov_rad), 0]],
            [[0, 0, 0], [0, max_r * np.cos(fov_rad), max_r * np.sin(fov_rad) * 0.5]],
            [[0, 0, 0], [0, max_r * np.cos(fov_rad), -max_r * np.sin(fov_rad) * 0.5]],
        ])
        for line in fov_lines:
            fov_item = gl.GLLinePlotItem(
                pos=line, 
                color=(0, 0.8, 0.8, 0.25), 
                width=1.5
            )
            self.view.addItem(fov_item)
        
        # Range rings on ground
        for r in [1, 2, 3, 4, 5, 6]:
            theta = np.linspace(-np.pi/2, np.pi/2, 40)
            ring = np.column_stack([
                r * np.sin(theta),
                r * np.cos(theta),
                np.zeros_like(theta)
            ])
            color = (0.3, 0.3, 0.35, 0.6) if r <= 3 else (0.2, 0.2, 0.25, 0.4)
            ring_item = gl.GLLinePlotItem(pos=ring, color=color, width=1)
            self.view.addItem(ring_item)
        
        # Main scatter plot for tracked objects
        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), 
            color=(1, 0, 0, 0), 
            size=0
        )
        self.view.addItem(self.scatter)
        
        # Static objects scatter (separate for different styling)
        self._static_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(0.5, 0.5, 0.5, 0),
            size=0,
            pxMode=True
        )
        self.view.addItem(self._static_scatter)
    
    def _setup_timer(self):
        """Set up the update timer."""
        update_ms = int(1000 / self.config.update_rate)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(update_ms)
    
    def update_display(self):
        """
        Update the display with latest world state.
        
        Called by Qt timer, consumes latest WorldState from queue.
        """
        # Drain queue, keep only most recent state
        latest_state = None
        try:
            while True:
                latest_state = self.state_queue.get_nowait()
        except queue.Empty:
            pass
        
        if latest_state is None:
            if self._last_state is None:
                self._update_info_no_data()
            return
        
        self._last_state = latest_state
        
        # Track FPS
        now = time.time()
        self._fps_times.append(now)
        
        # Clear dynamic elements
        self._clear_dynamic_elements()
        
        # Render tracks
        self._render_tracks(latest_state.tracks)
        
        # Render static objects
        self._render_static_objects(latest_state.static_objects)

        # Update info panel
        self._update_info(latest_state)

        # Update video playback if available
        self._update_video(latest_state.timestamp)
    
    def _clear_dynamic_elements(self):
        """Remove dynamic visualization elements."""
        for item in self._vert_lines:
            self.view.removeItem(item)
        self._vert_lines = []
        
        for item in self._velocity_arrows:
            self.view.removeItem(item)
        self._velocity_arrows = []
        
        for item in self._trail_items:
            self.view.removeItem(item)
        self._trail_items = []
    
    def _render_tracks(self, tracks: List[Track]):
        """Render tracked objects."""
        if not tracks:
            self.scatter.setData(
                pos=np.zeros((1, 3)),
                color=(1, 0, 0, 0),
                size=0
            )
            return

        # Prepare scatter data
        positions = np.array([[t.x, t.y, t.z] for t in tracks])
        colors = []
        sizes = []

        for t in tracks:
            # Color by threat level for approaching targets, otherwise by classification
            if t.classification == ObjectClass.STATIC:
                color = (0.6, 0.6, 0.7, 0.9)
            elif t.misses > 0:
                # Coasting - fade to purple/magenta to indicate predicted position
                fade = max(0.3, 1.0 - t.misses * 0.1)
                color = (0.8, 0.3, 0.8, fade)  # Magenta, fading
            elif t.is_threat:
                # Threat coloring based on threat score
                threat = t.threat_score / 100.0
                # Red/orange gradient for threats
                color = (1.0, 0.3 * (1 - threat), 0.0, 0.95)
            elif t.is_approaching:
                # Yellow/orange for approaching but not yet threatening
                color = (1.0, 0.7, 0.2, 0.9)
            else:
                color = color_by_range(t.range)
            colors.append(color)

            # Size based on confidence and threat, smaller when coasting
            base_size = 12
            if t.misses > 0:
                size = max(6, base_size - t.misses)  # Shrink while coasting
            elif t.is_threat:
                size = base_size + 8 + int(t.threat_score / 10)  # Larger for threats
            else:
                size = base_size + int(t.confidence * 8)
            sizes.append(size)
        
        colors = np.array(colors)
        sizes = np.array(sizes)
        
        self.scatter.setData(pos=positions, color=colors, size=sizes)
        
        # Add vertical lines for depth perception
        for t in tracks:
            if abs(t.z) > 0.05:  # Only if not on ground
                line_pos = np.array([
                    [t.x, t.y, 0],
                    [t.x, t.y, t.z]
                ])
                line = gl.GLLinePlotItem(
                    pos=line_pos, 
                    color=(0.4, 0.4, 0.5, 0.4), 
                    width=1
                )
                self.view.addItem(line)
                self._vert_lines.append(line)
        
        # Add velocity vectors for dynamic objects
        for t in tracks:
            if t.classification == ObjectClass.DYNAMIC and t.velocity_magnitude > 0.1:
                self._render_velocity_arrow(t)
        
        # Add trails
        for t in tracks:
            if len(t.history) > 1:
                self._render_trail(t)
    
    def _render_velocity_arrow(self, track: Track):
        """Render velocity vector as an arrow."""
        # Scale velocity for visibility (1 m/s = 0.5m arrow)
        scale = 0.5
        arrow_pos = np.array([
            [track.x, track.y, track.z],
            [track.x + track.vx * scale, 
             track.y + track.vy * scale, 
             track.z + track.vz * scale]
        ])
        
        # Color based on speed
        speed = track.velocity_magnitude
        if speed > 1.5:
            color = (1, 0.3, 0.3, 0.9)  # Fast = red
        elif speed > 0.5:
            color = (1, 0.8, 0.3, 0.9)  # Medium = yellow
        else:
            color = (0.3, 1, 0.3, 0.9)  # Slow = green
        
        arrow = gl.GLLinePlotItem(
            pos=arrow_pos,
            color=color,
            width=2.5
        )
        self.view.addItem(arrow)
        self._velocity_arrows.append(arrow)
    
    def _render_trail(self, track: Track):
        """Render track history as a fading trail."""
        if len(track.history) < 2:
            return
        
        # Get recent history points
        history = track.history[-min(len(track.history), 30):]
        trail_pos = np.array([[h[0], h[1], h[2]] for h in history])
        
        # Create fading colors
        n = len(history)
        if track.classification == ObjectClass.STATIC:
            base_color = np.array([0.5, 0.5, 0.6])
        else:
            base_color = np.array([0.3, 0.8, 1.0])
        
        colors = np.zeros((n, 4))
        for i in range(n):
            alpha = 0.1 + 0.5 * (i / n)  # Fade from transparent to more visible
            colors[i] = [*base_color, alpha]
        
        trail = gl.GLLinePlotItem(
            pos=trail_pos,
            color=colors,
            width=1.5
        )
        self.view.addItem(trail)
        self._trail_items.append(trail)
    
    def _render_static_objects(self, static_objects: List[StaticObject]):
        """Render static objects as gray markers."""
        if not static_objects:
            self._static_scatter.setData(
                pos=np.zeros((1, 3)),
                color=(0.5, 0.5, 0.5, 0),
                size=0
            )
            return
        
        positions = np.array([[s.x, s.y, s.z] for s in static_objects])
        
        # Size based on detection count
        sizes = np.array([min(20, 8 + s.detections // 5) for s in static_objects])
        
        # Gray color for static
        colors = np.full((len(static_objects), 4), [0.5, 0.5, 0.6, 0.7])
        
        self._static_scatter.setData(
            pos=positions,
            color=colors,
            size=sizes
        )
    
    def _update_info(self, state: WorldState):
        """Update the info panel with track details."""
        # Calculate display FPS
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            display_fps = len(self._fps_times) / elapsed if elapsed > 0 else 0
        else:
            display_fps = 0

        health = state.radar_health

        # Header
        info = f"<span style='color:#00cccc;'>■</span> Frame: {state.frame_number}  "
        info += f"<span style='color:#88cc88;'>●</span> Radar FPS: {health.fps:.1f}  "
        info += f"<span style='color:#cccc88;'>●</span> Display FPS: {display_fps:.1f}  "
        info += f"<span style='color:#cc8888;'>▲</span> Tracks: {len(state.tracks)}  "
        info += f"<span style='color:#888888;'>◆</span> Static: {len(state.static_objects)}"
        info += "<br/>"

        # Threat summary - show highest threat prominently
        highest_threat = state.get_highest_threat()
        if highest_threat and highest_threat.threat_score > 10:
            threat_color = "#ff4444" if highest_threat.threat_score > 50 else "#ffaa44"
            tti = highest_threat.time_to_intercept
            tti_str = f"{tti:.1f}s" if tti < 100 else "---"
            cv = highest_threat.closing_velocity
            cv_str = f"{cv:+.1f}" if abs(cv) < 100 else "---"
            info += f"<span style='color:{threat_color};'>⚠ THREAT [{highest_threat.track_id}]</span> "
            info += f"Score:<span style='color:{threat_color};'>{highest_threat.threat_score:.0f}</span>  "
            info += f"TTI:<span style='color:{threat_color};'>{tti_str}</span>  "
            info += f"Cv:<span style='color:{threat_color};'>{cv_str}</span>m/s"
            info += "<br/>"
        else:
            info += "<span style='color:#446644;'>○ No active threats</span><br/>"

        info += "<span style='color:#444455;'>─" * 40 + "</span><br/>"

        # Track details (up to 5, leaving room for threat header)
        for i in range(5):
            if i < len(state.tracks):
                t = state.tracks[i]

                # Icon and color based on threat/status
                if t.classification == ObjectClass.STATIC:
                    cls_icon = "◆"
                    cls_color = "#888888"
                elif t.is_threat:
                    cls_icon = "⚠"
                    cls_color = "#ff6644"
                elif t.is_approaching:
                    cls_icon = "→"
                    cls_color = "#ffaa44"
                elif t.misses > 0:
                    cls_icon = "◌"  # Hollow circle for coasting
                    cls_color = "#cc66cc"  # Magenta
                else:
                    cls_icon = "●"
                    cls_color = "#00cccc"

                # Build track info line
                info += f"<span style='color:{cls_color};'>{cls_icon}</span> "
                info += f"<span style='color:#aaaaaa;'>[{t.track_id:3d}]</span> "
                info += f"<span style='color:#ffffff;'>{t.range:.2f}m</span>  "

                # Show closing velocity for dynamic tracks
                if t.classification != ObjectClass.STATIC:
                    cv = t.closing_velocity
                    if cv > 0.5:
                        cv_color = "#ff6644"  # Approaching fast
                    elif cv > 0.1:
                        cv_color = "#ffaa44"  # Approaching slow
                    elif cv < -0.1:
                        cv_color = "#44aa44"  # Receding
                    else:
                        cv_color = "#888888"  # Stationary
                    info += f"Cv:<span style='color:{cv_color};'>{cv:+.1f}</span>  "

                    # TTI for approaching targets
                    tti = t.time_to_intercept
                    if tti < 10:
                        tti_color = "#ff4444" if tti < 3 else "#ffaa44"
                        info += f"TTI:<span style='color:{tti_color};'>{tti:.1f}s</span>  "
                else:
                    info += f"Az:<span style='color:#88aacc;'>{t.azimuth_deg:+5.0f}°</span>  "

                # Threat score for non-static
                if t.classification != ObjectClass.STATIC and t.threat_score > 5:
                    ts_color = "#ff4444" if t.threat_score > 50 else "#ffaa44" if t.threat_score > 20 else "#888888"
                    info += f"T:<span style='color:{ts_color};'>{t.threat_score:.0f}</span>"

                info += "<br/>"
            else:
                info += f"<span style='color:#333344;'>  [---] ─────</span><br/>"

        if len(state.tracks) > 5:
            info += f"<span style='color:#666677;'>  ... and {len(state.tracks) - 5} more</span><br/>"

        self.info_label.setText(info)
    
    def _update_info_no_data(self):
        """Update info panel when no data is available."""
        info = "<span style='color:#cc6666;'>⚠</span> Waiting for radar data..."
        info += "<br/><span style='color:#444455;'>─" * 40 + "</span><br/>"
        for i in range(6):
            info += f"<span style='color:#333344;'>  [---] ─────</span><br/>"
        self.info_label.setText(info)
    
    def _update_video(self, timestamp: float):
        """Update video playback window if available."""
        if not self.video_player or not CV2_AVAILABLE:
            return

        frame = self.video_player.get_frame_at(timestamp)
        if frame is not None:
            if not self._video_window_created:
                cv2.namedWindow("Radar Video", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Radar Video", 640, 480)
                self._video_window_created = True

            cv2.imshow("Radar Video", frame)
            cv2.waitKey(1)  # Required for OpenCV to process window events

    def closeEvent(self, event):
        """Handle window close event."""
        self.timer.stop()
        # Close OpenCV window if it was created
        if self._video_window_created and CV2_AVAILABLE:
            cv2.destroyAllWindows()
        event.accept()

