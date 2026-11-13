"""Spatial Environment Module for HALO.

The module creates a 2D urban grid, emission sources, Gaussian
concentration fields, and complete simulation scenarios.
"""

from .types import SourceType, GridConfig, EmissionSource, EnvironmentResult
from .grid import Grid
from .sources import SourceFactory, create_sources
from .gaussian import GaussianDispersion
from .scenario import EnvironmentScenario, create_scenario

__all__ = [
    "SourceType",
    "GridConfig",
    "EmissionSource",
    "EnvironmentResult",
    "Grid",
    "SourceFactory",
    "create_sources",
    "GaussianDispersion",
    "EnvironmentScenario",
    "create_scenario",
]
