from __future__ import annotations

import numpy as np

from .types import GridConfig


class Grid:
    """2D cell-center coordinate system.

    This class contains spatial geometry only. It deliberately knows
    nothing about pollution, sources, Gaussian dispersion, or HARO.
    """

    def __init__(self, config: GridConfig) -> None:
        self.config = config

        x = config.origin_x + np.arange(config.width) * config.resolution
        y = config.origin_y + np.arange(config.height) * config.resolution

        self.x = x
        self.y = y
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    @property
    def n_cells(self) -> int:
        return self.config.width * self.config.height

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return xmin, xmax, ymin, ymax using cell-center coordinates."""
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.y[0]),
            float(self.y[-1]),
        )

    def contains(self, x: float, y: float) -> bool:
        xmin, xmax, ymin, ymax = self.bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def validate_point(self, x: float, y: float) -> None:
        if not self.contains(x, y):
            raise ValueError(
                f"Point ({x}, {y}) is outside grid bounds {self.bounds}."
            )

    def nearest_cell(self, x: float, y: float) -> tuple[int, int]:
        """Return (row, col) of the nearest grid cell."""
        self.validate_point(x, y)
        col = int(np.argmin(np.abs(self.x - x)))
        row = int(np.argmin(np.abs(self.y - y)))
        return row, col

    def flatten_coordinates(self) -> np.ndarray:
        """Return coordinates as [x, y] rows."""
        return np.column_stack((self.X.ravel(), self.Y.ravel()))
