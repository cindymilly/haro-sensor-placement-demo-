import numpy as np
import pytest

from environment.scenario import create_scenario
from environment.grid import Grid
from environment.types import GridConfig
from problem.candidates import CandidateSet
from problem.problem import MonitoringProblem

from .base import OptimizerConfig
from .aro import ARO
from .haro import HARO
from .pso import PSO
from .mvo import MVO
from .ga import GA


def make_problem():
    env = create_scenario(
        grid_size=(20, 20),
        sources=[
            {"x": 5, "y": 5, "source_type": "traffic", "emission_rate": 1.0},
            {"x": 14, "y": 14, "source_type": "industrial", "emission_rate": 2.0},
        ],
        sigma=4.0,
        background_noise_std=0.0,
    )

    grid = Grid(GridConfig(20, 20))
    candidates = CandidateSet.from_grid(grid)

    return MonitoringProblem(
        env,
        candidates,
        k=3,
        coverage_radius=3.0,
        min_station_distance=2.0,
    )


@pytest.mark.parametrize("optimizer_cls", [ARO, HARO, PSO, MVO, GA])
def test_optimizer_returns_feasible_solution(optimizer_cls):
    result = optimizer_cls(
        make_problem(),
        OptimizerConfig(population_size=8, max_iterations=5, seed=120, verbose=False),
    ).optimize()

    assert result.best_details["feasible"]
    assert result.best_solution.k == 3
    assert len(result.convergence) == 6


@pytest.mark.parametrize("optimizer_cls", [ARO, HARO, PSO, MVO, GA])
def test_reproducibility(optimizer_cls):
    config = OptimizerConfig(
        population_size=6, max_iterations=5, seed=120, verbose=False
    )
    r1 = optimizer_cls(make_problem(), config).optimize()
    r2 = optimizer_cls(make_problem(), config).optimize()

    assert np.isclose(r1.best_fitness, r2.best_fitness)
    assert np.array_equal(
        r1.best_solution.candidate_indices,
        r2.best_solution.candidate_indices,
    )
