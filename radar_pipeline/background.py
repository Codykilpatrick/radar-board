"""
Background Model - 3D voxel grid for background subtraction.

Learns static environment (walls, floor, furniture) and filters out
known clutter detections, allowing detection of new stationary objects.

Usage:
    # Learning phase
    model = BackgroundModel()
    for detection in detections:
        model.add_detection(detection.x, detection.y, detection.z)
    model.finalize(min_hits=3)
    model.save("background.npz")

    # Runtime filtering
    model = BackgroundModel.load("background.npz")
    if not model.is_background(x, y, z):
        # This is a real detection, not clutter
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class BackgroundModel:
    """
    3D voxel grid background model for radar clutter filtering.

    Divides the sensing volume into a 3D grid of cells. During learning,
    counts detections per cell. Cells with counts above threshold are
    marked as background. At runtime, detections in background cells
    are filtered out.
    """

    def __init__(self,
                 resolution: float = 0.1,
                 x_range: Tuple[float, float] = (-4.0, 4.0),
                 y_range: Tuple[float, float] = (0.0, 8.0),
                 z_range: Tuple[float, float] = (-1.0, 2.0)):
        """
        Initialize the background model.

        Args:
            resolution: Size of each voxel in meters (default 10cm)
            x_range: (min, max) X coordinates in meters
            y_range: (min, max) Y coordinates in meters (forward distance)
            z_range: (min, max) Z coordinates in meters (height)
        """
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        # Calculate grid dimensions
        self.nx = int(np.ceil((x_range[1] - x_range[0]) / resolution))
        self.ny = int(np.ceil((y_range[1] - y_range[0]) / resolution))
        self.nz = int(np.ceil((z_range[1] - z_range[0]) / resolution))

        # Voxel count grid (uint16 to save memory, max 65535 hits per cell)
        self._counts = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint16)

        # Background mask (True = background, filter out)
        self._mask: Optional[np.ndarray] = None

        # Learning stats
        self._total_detections = 0
        self._finalized = False

    def _to_grid_coords(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        ix = int((x - self.x_range[0]) / self.resolution)
        iy = int((y - self.y_range[0]) / self.resolution)
        iz = int((z - self.z_range[0]) / self.resolution)
        return ix, iy, iz

    def _in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        """Check if grid indices are within bounds."""
        return (0 <= ix < self.nx and
                0 <= iy < self.ny and
                0 <= iz < self.nz)

    def add_detection(self, x: float, y: float, z: float) -> bool:
        """
        Add a detection to the background model during learning.

        Args:
            x, y, z: Detection coordinates in meters

        Returns:
            True if detection was added, False if out of bounds
        """
        if self._finalized:
            return False

        ix, iy, iz = self._to_grid_coords(x, y, z)

        if not self._in_bounds(ix, iy, iz):
            return False

        # Increment count (saturating at max uint16)
        if self._counts[ix, iy, iz] < 65535:
            self._counts[ix, iy, iz] += 1

        self._total_detections += 1
        return True

    def finalize(self, min_hits: int = 3):
        """
        Finalize learning and create background mask.

        Args:
            min_hits: Minimum detections in a cell to mark as background
        """
        self._mask = self._counts >= min_hits
        self._finalized = True

        bg_cells = np.sum(self._mask)
        total_cells = self.nx * self.ny * self.nz

        print(f"[Background] Finalized: {bg_cells} background cells "
              f"({100*bg_cells/total_cells:.1f}% of volume)")
        print(f"[Background] Total detections processed: {self._total_detections}")

    def is_background(self, x: float, y: float, z: float) -> bool:
        """
        Check if a detection is in a background cell.

        Args:
            x, y, z: Detection coordinates in meters

        Returns:
            True if detection should be filtered (is background)
        """
        if self._mask is None:
            return False

        ix, iy, iz = self._to_grid_coords(x, y, z)

        if not self._in_bounds(ix, iy, iz):
            return False

        return bool(self._mask[ix, iy, iz])

    def save(self, filepath: str):
        """
        Save the background model to a file.

        Args:
            filepath: Path to save (will add .npz if not present)
        """
        path = Path(filepath)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')

        # Save both counts and mask for flexibility
        np.savez_compressed(
            path,
            counts=self._counts,
            mask=self._mask if self._mask is not None else np.zeros_like(self._counts, dtype=bool),
            resolution=self.resolution,
            x_range=self.x_range,
            y_range=self.y_range,
            z_range=self.z_range,
            total_detections=self._total_detections,
            finalized=self._finalized
        )

        print(f"[Background] Saved model to {path}")
        print(f"[Background] Grid size: {self.nx}x{self.ny}x{self.nz} = "
              f"{self._counts.nbytes / 1024:.1f} KB")

    @classmethod
    def load(cls, filepath: str) -> 'BackgroundModel':
        """
        Load a background model from file.

        Args:
            filepath: Path to .npz file

        Returns:
            Loaded BackgroundModel instance
        """
        path = Path(filepath)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')

        data = np.load(path)

        model = cls(
            resolution=float(data['resolution']),
            x_range=tuple(data['x_range']),
            y_range=tuple(data['y_range']),
            z_range=tuple(data['z_range'])
        )

        model._counts = data['counts']
        model._mask = data['mask']
        model._total_detections = int(data['total_detections'])
        model._finalized = bool(data['finalized'])

        bg_cells = np.sum(model._mask) if model._mask is not None else 0
        print(f"[Background] Loaded model from {path}")
        print(f"[Background] {bg_cells} background cells, "
              f"{model._total_detections} training detections")

        return model

    @staticmethod
    def exists(filepath: str) -> bool:
        """Check if a background model file exists."""
        path = Path(filepath)
        if path.suffix != '.npz':
            path = path.with_suffix('.npz')
        return path.exists()

    def get_stats(self) -> dict:
        """Get background model statistics."""
        bg_cells = np.sum(self._mask) if self._mask is not None else 0
        total_cells = self.nx * self.ny * self.nz

        return {
            'grid_size': (self.nx, self.ny, self.nz),
            'resolution': self.resolution,
            'total_cells': total_cells,
            'background_cells': int(bg_cells),
            'background_pct': 100 * bg_cells / total_cells if total_cells > 0 else 0,
            'training_detections': self._total_detections,
            'finalized': self._finalized,
            'memory_kb': self._counts.nbytes / 1024
        }
