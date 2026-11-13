from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .gaussian import GaussianDispersion
from .grid import Grid
from .sources import create_sources
from .types import EmissionSource, EnvironmentResult, GridConfig


class EnvironmentScenario:
    """Orchestrates Grid + Sources + Gaussian into one environment."""

    def __init__(
        self,
        grid: Grid,
        sources: Sequence[EmissionSource],
        gaussian: GaussianDispersion | None = None,
        scenario_name: str = "default",
        background_noise_std: float = 0.0,
        random_seed: int | None = None,
    ) -> None:
        if background_noise_std < 0:
            raise ValueError("background_noise_std must be >= 0.")

        self.grid = grid
        self.sources = list(sources)
        self.gaussian = gaussian or GaussianDispersion(sigma=8.0)
        self.scenario_name = scenario_name
        self.background_noise_std = float(background_noise_std)
        self.random_seed = random_seed

    def run(self) -> EnvironmentResult:
        concentration = self.gaussian.simulate(self.grid, self.sources)

        if self.background_noise_std > 0:
            rng = np.random.default_rng(self.random_seed)
            concentration = concentration + rng.normal(
                loc=0.0,
                scale=self.background_noise_std,
                size=self.grid.shape,
            )
            # Concentration cannot be negative.
            concentration = np.clip(concentration, 0.0, None)

        metadata = {
            "scenario_name": self.scenario_name,
            "grid": asdict(self.grid.config),
            "sigma_default": self.gaussian.sigma,
            "background_noise_std": self.background_noise_std,
            "random_seed": self.random_seed,
            "n_sources": len(self.sources),
            "concentration_min": float(concentration.min()),
            "concentration_max": float(concentration.max()),
            "concentration_mean": float(concentration.mean()),
        }

        return EnvironmentResult(
            concentration=concentration,
            grid_x=self.grid.X.copy(),
            grid_y=self.grid.Y.copy(),
            sources=self.sources.copy(),
            metadata=metadata,
        )


def create_scenario(
    grid_size: int | tuple[int, int],
    source_definitions: Sequence[dict | EmissionSource] | None = None,
    sigma: float = 8.0,
    resolution: float = 1.0,
    scenario_name: str = "default",
    background_noise_std: float = 0.0,
    random_seed: int | None = None,
    sources: Sequence[dict | EmissionSource] | None = None, # <--- Thêm tham số 'sources'
) -> EnvironmentResult:
    """Convenience function for creating and running a scenario."""
    
    # 1. Tự động hỗ trợ cả 'sources' lẫn 'source_definitions'
    defs = sources if sources is not None else source_definitions
    if defs is None:
        defs = []

    # 2. Xử lý grid_size nếu truyền dạng tuple (width, height)
    if isinstance(grid_size, (tuple, list)):
        width, height = grid_size
    else:
        width, height = grid_size, grid_size

    grid = Grid(
        GridConfig(
            width=width,
            height=height,
            resolution=resolution,
        )
    )

    sources_list = create_sources(defs)

    gaussian = GaussianDispersion(sigma=sigma)

    scenario = EnvironmentScenario(
        grid=grid,
        sources=sources_list,
        scenario_name=scenario_name,
        background_noise_std=background_noise_std,
        random_seed=random_seed,
    )

    return scenario.run()
