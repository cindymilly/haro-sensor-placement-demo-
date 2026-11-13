"""Run from project root after Block 01 is available:

    python -m problem.example
"""

import numpy as np
import matplotlib.pyplot as plt

from environment.scenario import create_scenario
from .candidates import CandidateSet
from .encoding import StationSolution
from .problem import MonitoringProblem


SOURCES = [
    {"x": 12, "y": 15, "source_type": "traffic", "emission_rate": 1.0},
    {"x": 35, "y": 38, "source_type": "industrial", "emission_rate": 2.0},
    {"x": 25, "y": 10, "source_type": "construction", "emission_rate": 1.5},
]


def main():
    scenario = create_scenario(
        grid_size=(50, 50),
        sources=SOURCES,
        sigma=8.0,
        scenario_name="block02_demo",
        background_noise_std=0.02,
        random_seed=120,
    )
    env = scenario.run()

    candidates = CandidateSet.from_grid(
        __import__("environment.grid", fromlist=["Grid"]).Grid(
            __import__("environment.types", fromlist=["GridConfig"]).GridConfig(50, 50)
        )
    )

    problem = MonitoringProblem(
        environment=env,
        candidates=candidates,
        k=5,
        coverage_radius=6.0,
        min_station_distance=5.0,
        penalty=10.0,
    )

    rng = np.random.default_rng(120)
    solution = problem.random_solution(rng)
    print("Problem:", problem.summary())
    print("Stations:", solution.candidate_indices.tolist())
    print("Coordinates:")
    print(solution.coordinates(candidates))
    print("Evaluation:")
    print(problem.evaluate_with_details(solution))

    mask = problem.objective.covered_mask(solution).reshape(env.concentration.shape)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        env.concentration,
        origin="lower",
        extent=[
            env.grid_x.min(), env.grid_x.max(),
            env.grid_y.min(), env.grid_y.max()
        ],
        aspect="auto",
    )
    coords = solution.coordinates(candidates)
    plt.scatter(coords[:, 0], coords[:, 1], marker="x", s=80)
    plt.contour(mask.astype(float), levels=[0.5], linewidths=1)
    plt.title("Block 02 – Monitoring Network Problem")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Concentration")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
