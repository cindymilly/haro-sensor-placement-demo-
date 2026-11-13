import numpy as np
import pytest

from environment.scenario import create_scenario
from environment.grid import Grid
from environment.types import GridConfig

from .candidates import CandidateSet
from .encoding import StationSolution
from .problem import MonitoringProblem


def make_problem(k=2, radius=3.0, min_distance=2.0):
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
        env, candidates, k=k,
        coverage_radius=radius,
        min_station_distance=min_distance,
    )


def test_k_constraint():
    problem = make_problem(k=3)
    rng = np.random.default_rng(120)
    sol = StationSolution.random(problem.candidates, 3, rng)
    assert sol.k == 3


def test_coverage_is_between_zero_and_one():
    problem = make_problem()
    sol = StationSolution(np.array([105, 294]))
    ratio = problem.objective.coverage_ratio(sol)
    assert 0.0 <= ratio <= 1.0


def test_duplicate_solution_is_infeasible():
    problem = make_problem()
    # StationSolution tự chặn dữ liệu trùng lặp ngay khi khởi tạo
    with pytest.raises(ValueError, match="cannot contain duplicate"):
        sol = StationSolution(np.array([10, 10]))


def test_distance_constraint():
    problem = make_problem(min_distance=5.0)
    sol = StationSolution(np.array([0, 1]))  # adjacent cells
    violation = problem.constraints.check(sol)
    assert not violation.feasible
    assert violation.distance_violation > 0


def test_fitness_is_negative_for_positive_coverage():
    problem = make_problem()
    sol = StationSolution(np.array([105, 294]))
    assert problem.objective.fitness(sol) <= 0.0
