from __future__ import annotations

import numpy as np

from .grid import Grid
from .types import EmissionSource


class GaussianDispersion:
    """Radial Gaussian concentration model.

    For source i:

        C_i(x,y) = A_i * exp(-d_i^2 / (2*sigma_i^2))

        d_i = sqrt((x-x_i)^2 + (y-y_i)^2)

    Total concentration is the sum over all sources.

    IMPORTANT:
    This is a distance-based radial Gaussian model. It is not a full
    atmospheric Gaussian plume model with wind direction, stack height,
    deposition, or chemical transformation.
    """

    def __init__(
        self,
        sigma: float = 8.0,
        include_source_type_scaling: bool = False,
        source_type_scaling: dict | None = None,
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")

        self.sigma = float(sigma)
        self.include_source_type_scaling = include_source_type_scaling

        self.source_type_scaling = source_type_scaling or {
            "traffic": 1.0,
            "industrial": 1.0,
            "construction": 1.0,
            "noise": 1.0,
        }

    @staticmethod
    def _distance_squared(X: np.ndarray, Y: np.ndarray, source: EmissionSource):
        return (X - source.x) ** 2 + (Y - source.y) ** 2

    def source_field(
        self,
        grid: Grid,
        source: EmissionSource,
    ) -> np.ndarray:
        """Compute the Gaussian field of one source."""
        grid.validate_point(source.x, source.y)

        sigma = source.sigma if source.sigma is not None else self.sigma
        d2 = self._distance_squared(grid.X, grid.Y, source)

        field = source.emission_rate * np.exp(-d2 / (2.0 * sigma**2))

        if self.include_source_type_scaling:
            factor = self.source_type_scaling[source.source_type.value]
            field = field * factor

        return field

    def simulate(
        self,
        grid: Grid,
        sources: list[EmissionSource],
    ) -> np.ndarray:
        """Return total concentration map."""
        concentration = np.zeros(grid.shape, dtype=float)

        for source in sources:
            concentration += self.source_field(grid, source)

        return concentration

    def simulate_by_source(
        self,
        grid: Grid,
        sources: list[EmissionSource],
    ) -> dict[str, np.ndarray]:
        """Return individual source fields keyed by source name."""
        fields: dict[str, np.ndarray] = {}

        for i, source in enumerate(sources):
            key = source.name or f"{source.source_type.value}_{i}"
            fields[key] = self.source_field(grid, source)

        return fields
