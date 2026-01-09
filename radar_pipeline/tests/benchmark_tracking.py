#!/usr/bin/env python3
"""
Benchmark tracking performance against ground truth.

Runs synthetic scenarios through the tracker and measures:
- Acquisition time
- Position/velocity accuracy
- Track continuity

Usage:
    python3 radar_pipeline/tests/benchmark_tracking.py [scenario]
    python3 radar_pipeline/tests/benchmark_tracking.py --all
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from radar_pipeline.config import PipelineConfig
from radar_pipeline.data_types import Detection
from radar_pipeline.scenarios import SCENARIOS
from radar_pipeline.metrics import MetricsCollector, GroundTruth, SessionMetrics
from radar_pipeline.tests.test_simulation import SimpleTracker


class BenchmarkScenario:
    """
    Wrapper that provides ground truth alongside detections.
    """

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.scenario = SCENARIOS[scenario_name]()

    def generate_with_truth(self, t: float) -> tuple:
        """
        Generate detections and corresponding ground truth.

        Returns:
            (detections, ground_truths) tuple
        """
        detections = self.scenario.generate(t)

        # Extract ground truth from scenario
        # Most scenarios have predictable motion we can compute
        ground_truths = self._compute_ground_truth(t)

        return detections, ground_truths

    def _compute_ground_truth(self, t: float) -> list:
        """Compute ground truth positions for known scenarios."""
        truths = []

        if self.scenario_name == 'forward':
            period = 8.0
            phase = (t % period) / period
            if phase < 0.5:
                y = 2.0 + 8.0 * phase
            else:
                y = 6.0 - 8.0 * (phase - 0.5)

            # Velocity
            if phase < 0.5:
                vy = 8.0 / (period / 2)  # 2 m/s forward
            else:
                vy = -8.0 / (period / 2)  # 2 m/s backward

            truths.append(GroundTruth(
                target_id=1, x=0.0, y=y, z=0.5,
                vx=0.0, vy=vy, vz=0.0, timestamp=t
            ))

        elif self.scenario_name == 'circular':
            radius = 2.0
            center_y = 4.0
            omega = 0.5

            x = radius * np.sin(omega * t)
            y = center_y + radius * np.cos(omega * t)
            z = 0.5 + 0.3 * np.sin(omega * t * 2)

            vx = radius * omega * np.cos(omega * t)
            vy = -radius * omega * np.sin(omega * t)
            vz = 0.3 * 2 * omega * np.cos(omega * t * 2)

            truths.append(GroundTruth(
                target_id=1, x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz, timestamp=t
            ))

        elif self.scenario_name == 'crossing':
            # Target 1: left to right
            x1 = -2.0 + 0.5 * (t % 16.0)
            if -3 < x1 < 3:
                truths.append(GroundTruth(
                    target_id=1, x=x1, y=3.0, z=0.4,
                    vx=0.5, vy=0.0, vz=0.0, timestamp=t
                ))

            # Target 2: right to left
            x2 = 2.0 - 0.5 * (t % 16.0)
            if -3 < x2 < 3:
                truths.append(GroundTruth(
                    target_id=2, x=x2, y=4.0, z=0.6,
                    vx=-0.5, vy=0.0, vz=0.0, timestamp=t
                ))

        elif self.scenario_name == 'dropout':
            y = 2.0 + 0.5 * (t % 12.0)
            truths.append(GroundTruth(
                target_id=1, x=0.0, y=min(y, 6.0), z=0.5,
                vx=0.0, vy=0.5, vz=0.0, timestamp=t
            ))

        elif self.scenario_name == 'approaching':
            period = 12.0
            phase = t % period
            if phase < 10.0:
                y = 7.0 - 0.6 * phase
                vy = -0.6
            else:
                y = 1.0
                vy = 0.0

            x = 0.3 * np.sin(t * 0.5)
            vx = 0.3 * 0.5 * np.cos(t * 0.5)

            truths.append(GroundTruth(
                target_id=1, x=x, y=y, z=0.5,
                vx=vx, vy=vy, vz=0.0, timestamp=t
            ))

        elif self.scenario_name == 'multi':
            # Near target
            x1 = 1.5 * np.sin(t * 0.8)
            vx1 = 1.5 * 0.8 * np.cos(t * 0.8)
            truths.append(GroundTruth(
                target_id=1, x=x1, y=2.0, z=0.4,
                vx=vx1, vy=0.0, vz=0.0, timestamp=t
            ))

            # Mid-range target
            x2 = 0.5 * np.sin(t * 0.3)
            y2 = 4.0 + 1.0 * np.sin(t * 0.2)
            vx2 = 0.5 * 0.3 * np.cos(t * 0.3)
            vy2 = 1.0 * 0.2 * np.cos(t * 0.2)
            truths.append(GroundTruth(
                target_id=2, x=x2, y=y2, z=0.6,
                vx=vx2, vy=vy2, vz=0.0, timestamp=t
            ))

            # Far target
            x3 = -0.5 + 0.2 * np.sin(t * 0.1)
            vx3 = 0.2 * 0.1 * np.cos(t * 0.1)
            truths.append(GroundTruth(
                target_id=3, x=x3, y=6.5, z=0.5,
                vx=vx3, vy=0.0, vz=0.0, timestamp=t
            ))

        else:
            # For scenarios without implemented ground truth,
            # return empty (metrics will just measure what we can)
            pass

        return truths


def run_benchmark(scenario_name: str, num_frames: int = 200, verbose: bool = True) -> SessionMetrics:
    """
    Run benchmark for a single scenario.

    Args:
        scenario_name: Name of scenario to benchmark
        num_frames: Number of frames to simulate
        verbose: Print detailed output

    Returns:
        SessionMetrics with results
    """
    config = PipelineConfig()
    tracker = SimpleTracker(config)
    collector = MetricsCollector(frame_period=config.frame_period)

    benchmark = BenchmarkScenario(scenario_name)

    if verbose:
        print(f"\nRunning benchmark: {scenario_name}")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: {num_frames * config.frame_period:.1f}s")

    np.random.seed(42)  # Reproducible

    for frame in range(num_frames):
        t = frame * config.frame_period

        # Generate detections and ground truth
        detections, ground_truths = benchmark.generate_with_truth(t)

        # Run tracker
        confirmed_tracks = tracker.process_detections(detections, t)

        # Record metrics
        collector.record_frame(frame, ground_truths, confirmed_tracks, detections)

    metrics = collector.compute_metrics()

    if verbose:
        collector.print_report(metrics)

    return metrics


def run_all_benchmarks(num_frames: int = 200) -> dict:
    """Run benchmarks on all scenarios with ground truth support."""
    supported_scenarios = ['forward', 'circular', 'crossing', 'dropout', 'approaching', 'multi']

    results = {}
    print("\n" + "=" * 70)
    print("TRACKING BENCHMARK SUITE")
    print("=" * 70)

    for scenario in supported_scenarios:
        print(f"\n{'─' * 70}")
        metrics = run_benchmark(scenario, num_frames, verbose=True)
        results[scenario] = metrics

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<12} {'Acq(ms)':<10} {'Pos(cm)':<10} {'Vel(m/s)':<10} {'Gaps':<8} {'False':<8}")
    print("-" * 70)

    for scenario, metrics in results.items():
        gaps = sum(tm.gaps for tm in metrics.track_metrics)
        print(f"{scenario:<12} "
              f"{metrics.mean_acquisition_time_ms:<10.0f} "
              f"{metrics.mean_position_error*100:<10.1f} "
              f"{metrics.mean_velocity_error:<10.2f} "
              f"{gaps:<8} "
              f"{metrics.false_tracks:<8}")

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark tracking performance")
    parser.add_argument('scenario', nargs='?', default=None,
                        help="Scenario to benchmark (or --all)")
    parser.add_argument('--all', '-a', action='store_true',
                        help="Run all benchmarks")
    parser.add_argument('--frames', '-n', type=int, default=200,
                        help="Number of frames to simulate (default: 200)")
    args = parser.parse_args()

    if args.all or args.scenario is None:
        run_all_benchmarks(args.frames)
    else:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            return 1
        run_benchmark(args.scenario, args.frames, verbose=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
