from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class SourceType(str, Enum):
    """Supported simulated source categories."""

    TRAFFIC = "traffic"
    INDUSTRIAL = "industrial"
    CONSTRUCTION = "construction"
    NOISE = "noise"


@dataclass(frozen=True)
class GridConfig:
    """Configuration of a rectangular 2D grid.

    width/height are the number of cells.
    resolution is the physical distance represented by one cell.
    """

    width: int
    height: int
    resolution: float = 1.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("Grid width and height must be >= 2.")
        if self.resolution <= 0:
            raise ValueError("Grid resolution must be > 0.")


@dataclass(frozen=True)
class EmissionSource:
    """A point emission source.

    emission_rate is the source amplitude used by the Gaussian model.
    sigma can override the scenario-level default sigma.
    """

    x: float
    y: float
    source_type: SourceType | str
    emission_rate: float = 1.0
    sigma: float | None = None
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_type = (
            self.source_type
            if isinstance(self.source_type, SourceType)
            else SourceType(str(self.source_type).lower())
        )
        object.__setattr__(self, "source_type", source_type)

        if self.emission_rate < 0:
            raise ValueError("emission_rate must be >= 0.")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("sigma must be > 0 when provided.")


@dataclass
class EnvironmentResult:
    """Output contract passed from 01_environment to later modules."""

    concentration: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    sources: list[EmissionSource]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.concentration.ndim != 2:
            raise ValueError("concentration must be a 2D array.")
        if self.grid_x.shape != self.concentration.shape:
            raise ValueError("grid_x and concentration must have the same shape.")
        if self.grid_y.shape != self.concentration.shape:
            raise ValueError("grid_y and concentration must have the same shape.")
