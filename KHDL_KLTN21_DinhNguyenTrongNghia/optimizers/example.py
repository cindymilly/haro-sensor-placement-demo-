"""Block 03 demo comparing all 5 algorithms (HARO, ARO, PSO, MVO, GA).

Run from the HALO project root:

    python -m optimizers.example
"""

import matplotlib.pyplot as plt
import numpy as np

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


def build_problem():
    scenario = create_scenario(
        grid_size=(50, 50),
        sources=[
            {"x": 12, "y": 15, "source_type": "traffic", "emission_rate": 1.0},
            {"x": 35, "y": 38, "source_type": "industrial", "emission_rate": 2.0},
            {"x": 25, "y": 10, "source_type": "construction", "emission_rate": 1.5},
        ],
        sigma=8.0,
        background_noise_std=0.02,
        random_seed=120,
    )

    grid = Grid(GridConfig(50, 50))
    candidates = CandidateSet.from_grid(grid)

    return MonitoringProblem(
        environment=scenario,
        candidates=candidates,
        k=5,
        coverage_radius=6.0,
        min_station_distance=5.0,
    )


def main():
    problem = build_problem()

    config = OptimizerConfig(
        population_size=20,
        max_iterations=50,
        seed=120,
        verbose=False,
    )

    algorithms = [
        ("HARO", HARO(problem, config)),
        ("ARO", ARO(problem, config)),
        ("PSO", PSO(problem, config)),
        ("MVO", MVO(problem, config)),
        ("GA", GA(problem, config)),
    ]

    results = []
    print("\n=== SO SÁNH 5 THUẬT TOÁN TỐI ƯU HÓA ===")
    for name, opt in algorithms:
        res = opt.optimize()
        results.append(res)
        print(f"\n[{res.algorithm}]")
        print(f"  Best fitness: {res.best_fitness:.6f}")
        print(f"  Coverage:     {res.best_details['coverage_percent']:.2f}%")
        print(f"  Feasible:     {res.best_details['feasible']}")
        print(f"  Stations:     {res.best_solution.candidate_indices.tolist()}")
        print(f"  Evaluations:  {res.evaluations}")
        print(f"  Runtime:      {res.runtime_seconds:.2f}s")

    plt.figure(figsize=(9, 5))
    for res in results:
        plt.plot(res.convergence, label=res.algorithm, linewidth=2.0 if res.algorithm == "HARO" else 1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (lower is better)")
    plt.title("Convergence Comparison: HARO vs ARO vs PSO vs MVO vs GA")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("demo_optimizers_convergence.png", dpi=150)
    print("\nĐã lưu đồ thị demo vào: demo_optimizers_convergence.png")


if __name__ == "__main__":
    main()
