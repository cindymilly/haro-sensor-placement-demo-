import pytest
import numpy as np

from environment.scenario import create_scenario
from environment.grid import Grid
from environment.types import GridConfig
from problem.candidates import CandidateSet
from problem.problem import MonitoringProblem, BudgetExhaustedException

from optimizers.base import OptimizerConfig
from optimizers.aro import ARO
from optimizers.haro import HARO
from optimizers.pso import PSO
from optimizers.mvo import MVO
from optimizers.ga import GA


def make_problem():
    env = create_scenario(
        grid_size=(15, 15),
        sources=[
            {"x": 4, "y": 4, "source_type": "traffic", "emission_rate": 1.0},
            {"x": 10, "y": 10, "source_type": "industrial", "emission_rate": 2.0},
        ],
        sigma=3.0,
        background_noise_std=0.0,
    )
    grid = Grid(GridConfig(15, 15))
    candidates = CandidateSet.from_grid(grid)
    return MonitoringProblem(
        env,
        candidates,
        k=3,
        coverage_radius=2.5,
        min_station_distance=2.0,
    )


@pytest.mark.parametrize("optimizer_cls", [ARO, HARO, PSO, MVO, GA])
def test_fe_budget_strict_enforcement(optimizer_cls):
    """Verify that no optimizer ever exceeds the allotted max_fes budget."""
    problem = make_problem()
    max_fes = 75  # odd number to test mid-generation stopping
    config = OptimizerConfig(
        population_size=10,
        max_iterations=20,
        max_fes=max_fes,
        seed=42,
        verbose=False,
    )
    optimizer = optimizer_cls(problem, config)
    res = optimizer.optimize()

    assert res.evaluations <= max_fes, f"{optimizer_cls.__name__} exceeded FE budget: {res.evaluations} > {max_fes}"
    assert res.evaluations == problem.fe_count
    assert res.best_details["feasible"] is True


@pytest.mark.parametrize("optimizer_cls", [ARO, HARO, PSO, MVO, GA])
def test_fe_exact_count_standard_budget(optimizer_cls):
    """Verify that under standard iteration budget, FE exactly equals N + G * N."""
    problem = make_problem()
    pop_size = 10
    max_iter = 5
    expected_fe = pop_size + max_iter * pop_size  # 60 evaluations
    config = OptimizerConfig(
        population_size=pop_size,
        max_iterations=max_iter,
        max_fes=expected_fe,
        seed=123,
        verbose=False,
    )
    optimizer = optimizer_cls(problem, config)
    res = optimizer.optimize()

    assert res.evaluations == expected_fe, f"{optimizer_cls.__name__} FE was {res.evaluations}, expected {expected_fe}"
    assert problem.fe_count == expected_fe
    assert res.metadata["actual_fe"] == expected_fe


def test_zero_fe_for_details_query():
    """Verify that reading solution details does not count against FE budget."""
    problem = make_problem()
    rng = np.random.default_rng(42)
    problem.reset_fe(max_fes=100)
    sol = problem.random_solution(rng)

    initial_fe = problem.fe_count
    val = problem.evaluate(sol, count_fe=True)
    assert problem.fe_count == initial_fe + 1

    # Detailed evaluation with count_fe=False must cost 0 FE
    details = problem.evaluate_with_details(sol, count_fe=False)
    assert problem.fe_count == initial_fe + 1
    assert "coverage_ratio" in details
    assert "total_violation" in details


def test_budget_exhausted_exception():
    """Verify that calling evaluate after budget exhaustion raises BudgetExhaustedException."""
    problem = make_problem()
    rng = np.random.default_rng(42)
    problem.reset_fe(max_fes=2)
    sol = problem.random_solution(rng)

    problem.evaluate(sol, count_fe=True)
    problem.evaluate(sol, count_fe=True)
    assert problem.fe_count == 2

    with pytest.raises(BudgetExhaustedException):
        problem.evaluate(sol, count_fe=True)
