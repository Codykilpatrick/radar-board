"""
Tracking Metrics - Measure and analyze tracking performance.

Provides ground-truth comparison for synthetic scenarios and
computes key metrics for CIWS-style tracking:
- Acquisition time (frames to confirmed track)
- Position/velocity accuracy
- Track continuity
- Threat detection latency
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .data_types import Track, Detection


@dataclass
class GroundTruth:
    """Ground truth for a synthetic target."""
    target_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    timestamp: float


@dataclass
class TrackMetrics:
    """Metrics for a single track's performance."""
    track_id: int
    target_id: int  # Associated ground truth target

    # Acquisition
    first_detection_frame: int = 0
    first_confirmed_frame: int = 0
    acquisition_time_frames: int = 0
    acquisition_time_ms: float = 0.0

    # Accuracy (RMS errors)
    position_error_rms: float = 0.0
    velocity_error_rms: float = 0.0

    # Individual axis errors
    x_error_rms: float = 0.0
    y_error_rms: float = 0.0
    z_error_rms: float = 0.0
    vx_error_rms: float = 0.0
    vy_error_rms: float = 0.0
    vz_error_rms: float = 0.0

    # Continuity
    total_frames: int = 0
    frames_tracked: int = 0
    gaps: int = 0  # Number of times track was lost and reacquired
    longest_gap: int = 0

    # Threat detection
    closing_velocity_error_rms: float = 0.0


@dataclass
class SessionMetrics:
    """Aggregate metrics for an entire tracking session."""
    total_frames: int = 0
    total_targets: int = 0
    total_tracks: int = 0

    # Acquisition stats
    mean_acquisition_time_frames: float = 0.0
    mean_acquisition_time_ms: float = 0.0
    max_acquisition_time_frames: int = 0
    targets_never_acquired: int = 0

    # Accuracy stats
    mean_position_error: float = 0.0
    mean_velocity_error: float = 0.0

    # False tracks
    false_tracks: int = 0  # Tracks not matching any ground truth

    # Per-track metrics
    track_metrics: List[TrackMetrics] = field(default_factory=list)


class MetricsCollector:
    """
    Collects tracking metrics by comparing against ground truth.

    Usage:
        collector = MetricsCollector(frame_period=0.05)

        for frame in range(num_frames):
            # Get ground truth for this frame
            truths = [GroundTruth(target_id=1, x=..., y=..., ...)]

            # Get tracks from tracker
            tracks = tracker.process_detections(detections, t)

            # Record
            collector.record_frame(frame, truths, tracks)

        # Get results
        metrics = collector.compute_metrics()
        collector.print_report()
    """

    def __init__(self, frame_period: float = 0.05, association_threshold: float = 0.5):
        """
        Initialize metrics collector.

        Args:
            frame_period: Time between frames in seconds
            association_threshold: Max distance to associate track with ground truth
        """
        self.frame_period = frame_period
        self.association_threshold = association_threshold

        # Per-frame records
        self.frames: List[dict] = []

        # Track-to-target associations (track_id -> target_id)
        self.associations: Dict[int, int] = {}

        # First detection frame for each target
        self.target_first_detection: Dict[int, int] = {}

        # First confirmed track frame for each target
        self.target_first_confirmed: Dict[int, int] = {}

        # Track continuity: target_id -> list of (start_frame, end_frame)
        self.track_segments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._current_segments: Dict[int, int] = {}  # target_id -> start_frame

        # Error accumulation: target_id -> list of errors
        self.position_errors: Dict[int, List[float]] = defaultdict(list)
        self.velocity_errors: Dict[int, List[float]] = defaultdict(list)
        self.axis_errors: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Track all unique targets and tracks seen
        self.all_targets: set = set()
        self.all_tracks: set = set()
        self.false_track_ids: set = set()

    def record_frame(self,
                     frame_num: int,
                     ground_truths: List[GroundTruth],
                     tracks: List[Track],
                     detections: Optional[List[Detection]] = None):
        """
        Record a single frame for metrics computation.

        Args:
            frame_num: Frame number
            ground_truths: List of ground truth target states
            tracks: List of confirmed tracks from tracker
            detections: Optional raw detections (for detection-level metrics)
        """
        self.frames.append({
            'frame': frame_num,
            'truths': ground_truths,
            'tracks': tracks,
            'detections': detections or []
        })

        # Record all targets
        for gt in ground_truths:
            self.all_targets.add(gt.target_id)

            # Check if any detection is close to this target
            if detections:
                for det in detections:
                    dist = np.sqrt((det.x - gt.x)**2 + (det.y - gt.y)**2 + (det.z - gt.z)**2)
                    if dist < self.association_threshold:
                        if gt.target_id not in self.target_first_detection:
                            self.target_first_detection[gt.target_id] = frame_num
                        break

        # Associate tracks with ground truths
        for track in tracks:
            self.all_tracks.add(track.track_id)

            # Find closest ground truth
            best_gt = None
            best_dist = self.association_threshold

            for gt in ground_truths:
                dist = np.sqrt((track.x - gt.x)**2 + (track.y - gt.y)**2 + (track.z - gt.z)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_gt = gt

            if best_gt is not None:
                target_id = best_gt.target_id

                # Record association
                if track.track_id not in self.associations:
                    self.associations[track.track_id] = target_id

                # Record first confirmed frame
                if target_id not in self.target_first_confirmed:
                    self.target_first_confirmed[target_id] = frame_num

                # Track segment continuity
                if target_id not in self._current_segments:
                    self._current_segments[target_id] = frame_num

                # Record errors
                pos_err = np.sqrt((track.x - best_gt.x)**2 +
                                  (track.y - best_gt.y)**2 +
                                  (track.z - best_gt.z)**2)
                vel_err = np.sqrt((track.vx - best_gt.vx)**2 +
                                  (track.vy - best_gt.vy)**2 +
                                  (track.vz - best_gt.vz)**2)

                self.position_errors[target_id].append(pos_err)
                self.velocity_errors[target_id].append(vel_err)

                self.axis_errors[target_id]['x'].append(track.x - best_gt.x)
                self.axis_errors[target_id]['y'].append(track.y - best_gt.y)
                self.axis_errors[target_id]['z'].append(track.z - best_gt.z)
                self.axis_errors[target_id]['vx'].append(track.vx - best_gt.vx)
                self.axis_errors[target_id]['vy'].append(track.vy - best_gt.vy)
                self.axis_errors[target_id]['vz'].append(track.vz - best_gt.vz)
            else:
                # Track doesn't match any ground truth - false track
                self.false_track_ids.add(track.track_id)

        # Check for gaps (targets that were tracked but aren't now)
        tracked_targets = set()
        for track in tracks:
            if track.track_id in self.associations:
                tracked_targets.add(self.associations[track.track_id])

        for target_id in list(self._current_segments.keys()):
            if target_id not in tracked_targets:
                # Gap started - end current segment
                start = self._current_segments.pop(target_id)
                self.track_segments[target_id].append((start, frame_num - 1))

    def compute_metrics(self) -> SessionMetrics:
        """Compute aggregate metrics from recorded frames."""
        if not self.frames:
            return SessionMetrics()

        # Close any open segments
        last_frame = self.frames[-1]['frame']
        for target_id, start in self._current_segments.items():
            self.track_segments[target_id].append((start, last_frame))

        metrics = SessionMetrics(
            total_frames=len(self.frames),
            total_targets=len(self.all_targets),
            total_tracks=len(self.all_tracks),
            false_tracks=len(self.false_track_ids)
        )

        # Compute per-target metrics
        acquisition_times = []

        for target_id in self.all_targets:
            track_metric = TrackMetrics(
                track_id=0,  # Will be filled if we find associated track
                target_id=target_id
            )

            # Find associated track
            for track_id, assoc_target in self.associations.items():
                if assoc_target == target_id:
                    track_metric.track_id = track_id
                    break

            # Acquisition time
            if target_id in self.target_first_detection:
                track_metric.first_detection_frame = self.target_first_detection[target_id]

            if target_id in self.target_first_confirmed:
                track_metric.first_confirmed_frame = self.target_first_confirmed[target_id]

                if target_id in self.target_first_detection:
                    acq_frames = (track_metric.first_confirmed_frame -
                                  track_metric.first_detection_frame)
                    track_metric.acquisition_time_frames = acq_frames
                    track_metric.acquisition_time_ms = acq_frames * self.frame_period * 1000
                    acquisition_times.append(acq_frames)
            else:
                metrics.targets_never_acquired += 1

            # Position/velocity errors
            if target_id in self.position_errors and self.position_errors[target_id]:
                errors = self.position_errors[target_id]
                track_metric.position_error_rms = np.sqrt(np.mean(np.array(errors)**2))
                track_metric.frames_tracked = len(errors)

            if target_id in self.velocity_errors and self.velocity_errors[target_id]:
                errors = self.velocity_errors[target_id]
                track_metric.velocity_error_rms = np.sqrt(np.mean(np.array(errors)**2))

            # Axis errors
            if target_id in self.axis_errors:
                for axis in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
                    if self.axis_errors[target_id][axis]:
                        errors = self.axis_errors[target_id][axis]
                        rms = np.sqrt(np.mean(np.array(errors)**2))
                        setattr(track_metric, f'{axis}_error_rms', rms)

            # Continuity
            segments = self.track_segments.get(target_id, [])
            track_metric.gaps = max(0, len(segments) - 1)

            if segments:
                gap_lengths = []
                for i in range(1, len(segments)):
                    gap = segments[i][0] - segments[i-1][1] - 1
                    gap_lengths.append(gap)
                if gap_lengths:
                    track_metric.longest_gap = max(gap_lengths)

            track_metric.total_frames = len(self.frames)

            metrics.track_metrics.append(track_metric)

        # Aggregate stats
        if acquisition_times:
            metrics.mean_acquisition_time_frames = np.mean(acquisition_times)
            metrics.mean_acquisition_time_ms = metrics.mean_acquisition_time_frames * self.frame_period * 1000
            metrics.max_acquisition_time_frames = max(acquisition_times)

        position_errors_all = []
        velocity_errors_all = []
        for target_id in self.all_targets:
            position_errors_all.extend(self.position_errors.get(target_id, []))
            velocity_errors_all.extend(self.velocity_errors.get(target_id, []))

        if position_errors_all:
            metrics.mean_position_error = np.mean(position_errors_all)
        if velocity_errors_all:
            metrics.mean_velocity_error = np.mean(velocity_errors_all)

        return metrics

    def print_report(self, metrics: Optional[SessionMetrics] = None):
        """Print a human-readable metrics report."""
        if metrics is None:
            metrics = self.compute_metrics()

        print("\n" + "=" * 60)
        print("TRACKING METRICS REPORT")
        print("=" * 60)

        print(f"\nSession Overview:")
        print(f"  Total frames:     {metrics.total_frames}")
        print(f"  Total targets:    {metrics.total_targets}")
        print(f"  Total tracks:     {metrics.total_tracks}")
        print(f"  False tracks:     {metrics.false_tracks}")

        print(f"\nAcquisition Performance:")
        print(f"  Mean acquisition: {metrics.mean_acquisition_time_frames:.1f} frames "
              f"({metrics.mean_acquisition_time_ms:.0f} ms)")
        print(f"  Max acquisition:  {metrics.max_acquisition_time_frames} frames")
        print(f"  Never acquired:   {metrics.targets_never_acquired}")

        print(f"\nAccuracy (RMS):")
        print(f"  Position error:   {metrics.mean_position_error*100:.1f} cm")
        print(f"  Velocity error:   {metrics.mean_velocity_error:.2f} m/s")

        if metrics.track_metrics:
            print(f"\nPer-Target Details:")
            print("-" * 60)
            for tm in metrics.track_metrics:
                print(f"  Target {tm.target_id}:")
                print(f"    Acquisition:  {tm.acquisition_time_frames} frames ({tm.acquisition_time_ms:.0f} ms)")
                print(f"    Position RMS: {tm.position_error_rms*100:.1f} cm")
                print(f"    Velocity RMS: {tm.velocity_error_rms:.2f} m/s")
                print(f"    Gaps:         {tm.gaps} (longest: {tm.longest_gap} frames)")

        print("=" * 60)
