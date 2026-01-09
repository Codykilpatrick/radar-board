"""
Synthetic Radar Scenarios - Test patterns for development without hardware.

Each scenario generates Detection objects simulating radar returns.
Use with SyntheticDataSource for offline testing.
"""

import numpy as np
from typing import List
from abc import ABC, abstractmethod

from .data_types import Detection


class Scenario(ABC):
    """Base class for synthetic radar scenarios."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the scenario."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the scenario tests."""
        pass

    @abstractmethod
    def generate(self, t: float) -> List[Detection]:
        """
        Generate detections for the given time.

        Args:
            t: Time in seconds since start

        Returns:
            List of Detection objects
        """
        pass

    def _add_noise(self, x: float, y: float, z: float, noise: float = 0.03) -> tuple:
        """Add measurement noise to coordinates."""
        return (
            x + np.random.randn() * noise,
            y + np.random.randn() * noise,
            z + np.random.randn() * noise
        )

    def _make_detection(self, x: float, y: float, z: float, t: float,
                        snr: float = 20.0, noise: float = 0.03) -> Detection:
        """Create a Detection with noise applied."""
        x, y, z = self._add_noise(x, y, z, noise)
        return Detection(x=x, y=y, z=z, snr=snr, timestamp=t)


class ForwardBackScenario(Scenario):
    """Single target moving forward then back."""

    @property
    def name(self) -> str:
        return "forward"

    @property
    def description(self) -> str:
        return "Single target moving forward (y=2-6m) then back"

    def generate(self, t: float) -> List[Detection]:
        period = 8.0
        phase = (t % period) / period

        if phase < 0.5:
            y = 2.0 + 8.0 * phase
        else:
            y = 6.0 - 8.0 * (phase - 0.5)

        return [self._make_detection(0.0, y, 0.5, t)]


class CircularScenario(Scenario):
    """Target moving in a circle."""

    @property
    def name(self) -> str:
        return "circular"

    @property
    def description(self) -> str:
        return "Target orbiting in a circle (radius=2m, center at y=4m)"

    def generate(self, t: float) -> List[Detection]:
        radius = 2.0
        center_y = 4.0
        omega = 0.5

        x = radius * np.sin(omega * t)
        y = center_y + radius * np.cos(omega * t)
        z = 0.5 + 0.3 * np.sin(omega * t * 2)

        return [self._make_detection(x, y, z, t)]


class CrossingScenario(Scenario):
    """Two targets crossing paths."""

    @property
    def name(self) -> str:
        return "crossing"

    @property
    def description(self) -> str:
        return "Two targets crossing paths (tests track identity)"

    def generate(self, t: float) -> List[Detection]:
        # Target 1: moves left to right
        x1 = -2.0 + 0.5 * (t % 16.0)
        y1 = 3.0
        z1 = 0.4

        # Target 2: moves right to left
        x2 = 2.0 - 0.5 * (t % 16.0)
        y2 = 4.0
        z2 = 0.6

        detections = []

        if -3 < x1 < 3:
            detections.append(self._make_detection(x1, y1, z1, t, snr=20.0))

        if -3 < x2 < 3:
            detections.append(self._make_detection(x2, y2, z2, t, snr=18.0))

        return detections


class DropoutScenario(Scenario):
    """Target with periodic dropouts to test coasting."""

    @property
    def name(self) -> str:
        return "dropout"

    @property
    def description(self) -> str:
        return "Target with periodic dropouts (tests track coasting)"

    def generate(self, t: float) -> List[Detection]:
        y = 2.0 + 0.5 * (t % 12.0)

        # Drop out for 0.5s every 2s
        cycle_phase = t % 2.0
        if 1.0 < cycle_phase < 1.5:
            return []

        return [self._make_detection(0.0, min(y, 6.0), 0.5, t)]


class VerticalScenario(Scenario):
    """Target moving up and down (Z test)."""

    @property
    def name(self) -> str:
        return "vertical"

    @property
    def description(self) -> str:
        return "Target oscillating vertically (tests Z tracking)"

    def generate(self, t: float) -> List[Detection]:
        z = 0.5 + 0.8 * np.sin(0.5 * t)
        y = 3.0 + 0.3 * (t % 10.0)

        return [self._make_detection(0.0, min(y, 5.0), z, t)]


class RandomWalkScenario(Scenario):
    """Random walk stress test."""

    @property
    def name(self) -> str:
        return "random"

    @property
    def description(self) -> str:
        return "Erratic random motion (stress test for tracker)"

    def generate(self, t: float) -> List[Detection]:
        np.random.seed(int(t * 20) % 10000)

        x = np.sin(t * 0.2) * 1.5
        y = 3.0 + np.cos(t * 0.15) * 1.0
        z = 0.5 + np.sin(t * 0.3) * 0.3

        return [self._make_detection(x, y, z, t, noise=0.1)]


class MultiTargetScenario(Scenario):
    """Multiple targets at different ranges."""

    @property
    def name(self) -> str:
        return "multi"

    @property
    def description(self) -> str:
        return "3-4 targets at different ranges moving independently"

    def generate(self, t: float) -> List[Detection]:
        detections = []

        # Near target - fast lateral movement
        x1 = 1.5 * np.sin(t * 0.8)
        y1 = 2.0
        z1 = 0.4
        detections.append(self._make_detection(x1, y1, z1, t, snr=25.0))

        # Mid-range target - slow forward/back
        x2 = 0.5 * np.sin(t * 0.3)
        y2 = 4.0 + 1.0 * np.sin(t * 0.2)
        z2 = 0.6
        detections.append(self._make_detection(x2, y2, z2, t, snr=18.0))

        # Far target - mostly stationary with drift
        x3 = -0.5 + 0.2 * np.sin(t * 0.1)
        y3 = 6.5
        z3 = 0.5
        detections.append(self._make_detection(x3, y3, z3, t, snr=12.0))

        # Intermittent target - appears/disappears
        if np.sin(t * 0.4) > 0.3:
            x4 = 1.0
            y4 = 3.5 + 0.5 * np.sin(t * 0.5)
            z4 = 0.3
            detections.append(self._make_detection(x4, y4, z4, t, snr=15.0))

        return detections


class StaticClutterScenario(Scenario):
    """Static objects with one moving target."""

    @property
    def name(self) -> str:
        return "clutter"

    @property
    def description(self) -> str:
        return "Static clutter with one moving target (tests static classification)"

    def __init__(self):
        # Fixed static object positions
        self.static_objects = [
            (-1.5, 3.0, 0.3),
            (1.2, 4.5, 0.4),
            (0.0, 5.5, 0.2),
            (-0.8, 2.5, 0.5),
        ]

    def generate(self, t: float) -> List[Detection]:
        detections = []

        # Static objects with small noise
        for x, y, z in self.static_objects:
            detections.append(self._make_detection(x, y, z, t, snr=15.0, noise=0.02))

        # Moving target weaving through
        x_mov = 1.0 * np.sin(t * 0.4)
        y_mov = 2.5 + 1.5 * (t % 8.0) / 8.0 * 3.0  # moves from 2.5 to 5.5
        z_mov = 0.6

        if (t % 8.0) < 7.5:  # brief reset period
            detections.append(self._make_detection(x_mov, y_mov, z_mov, t, snr=22.0))

        return detections


class ApproachingScenario(Scenario):
    """Target approaching the radar directly."""

    @property
    def name(self) -> str:
        return "approaching"

    @property
    def description(self) -> str:
        return "Target approaching from far to near (tests range tracking)"

    def generate(self, t: float) -> List[Detection]:
        # Approach from y=7 to y=1 over 10 seconds, then reset
        period = 12.0
        phase = t % period

        if phase < 10.0:
            y = 7.0 - 0.6 * phase
        else:
            return []  # brief pause before reset

        # Slight lateral drift
        x = 0.3 * np.sin(t * 0.5)
        z = 0.5

        return [self._make_detection(x, y, z, t)]


# Registry of all available scenarios
SCENARIOS = {
    'forward': ForwardBackScenario,
    'circular': CircularScenario,
    'crossing': CrossingScenario,
    'dropout': DropoutScenario,
    'vertical': VerticalScenario,
    'random': RandomWalkScenario,
    'multi': MultiTargetScenario,
    'clutter': StaticClutterScenario,
    'approaching': ApproachingScenario,
}


def list_scenarios() -> dict:
    """
    Get all available scenarios with descriptions.

    Returns:
        Dict mapping scenario name to description
    """
    result = {}
    for name, cls in SCENARIOS.items():
        instance = cls()
        result[name] = instance.description
    return result
