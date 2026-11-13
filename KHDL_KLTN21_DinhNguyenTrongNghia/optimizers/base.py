from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import time
import numpy as np

from problem.problem import MonitoringProblem, BudgetExhaustedException
from problem.encoding import StationSolution


@dataclass(frozen=True)
class OptimizerConfig:
    population_size: int = 30
    max_iterations: int = 100
    max_fes: Optional[int] = None
    seed: int | None = 120
    verbose: bool = True

    def __post_init__(self):
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if self.max_fes is not None and self.max_fes < self.population_size:
            raise ValueError("max_fes cannot be less than population_size.")


@dataclass
class AgentState:
    """Runtime state of one optimizer agent."""

    vector: np.ndarray
    solution: StationSolution
    fitness: float
    feasible: bool
    energy: float = 0.0
    phase: str = "exploration"
    stagnation: int = 0
    halo_layer: str = "none"


@dataclass
class OptimizationResult:
    best_solution: StationSolution
    best_fitness: float
    best_details: dict
    convergence: np.ndarray
    evaluations: int
    runtime_seconds: float
    algorithm: str
    seed: int | None
    fe_convergence: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


class BaseOptimizer:
    """Common infrastructure shared by ARO, HARO, PSO, MVO, GA.

    Enforces strict centralized Function Evaluation (FE) budget accounting.
    No objective evaluation bypasses the problem counter.
    """

    def __init__(self, problem: MonitoringProblem, config: OptimizerConfig):
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Reset and bind problem centralized FE counter
        effective_max_fes = config.max_fes
        if effective_max_fes is None:
            # Default budget = N + G * N evaluations
            effective_max_fes = config.population_size + config.max_iterations * config.population_size
        self.problem.reset_fe(max_fes=effective_max_fes)
        self.evaluations = 0

        if getattr(problem, "is_continuous", False):
            self.is_continuous = True
            self.dimension = problem.dimension
            self.lb_vec = np.asarray(problem.lb, dtype=float)
            self.ub_vec = np.asarray(problem.ub, dtype=float)
            self.lb = self.lb_vec
            self.ub = self.ub_vec
            self.domain_diagonal = float(np.linalg.norm(self.ub_vec - self.lb_vec))
        else:
            self.is_continuous = False
            # Each station is represented by (x, y), so dimension = 2K.
            self.dimension = 2 * problem.k

            coords = problem.candidates.coordinates
            self.lb = np.min(coords, axis=0)
            self.ub = np.max(coords, axis=0)
            self.lb_vec = np.tile(self.lb, self.problem.k)
            self.ub_vec = np.tile(self.ub, self.problem.k)
            self.domain_diagonal = float(np.linalg.norm(self.ub - self.lb))

    def _budget_exhausted(self) -> bool:
        """Check whether the active FE budget is exhausted."""
        if self.config.max_fes is not None:
            return self.problem.fe_count >= self.config.max_fes
        return False

    def _evaluate_vector(self, vector: np.ndarray):
        """Decode continuous vector, evaluate through problem gateway, and update FE counter."""
        if self.is_continuous:
            v = np.clip(vector, self.lb_vec, self.ub_vec)
            fitness = float(self.problem.evaluate(v, count_fe=True))
            self.evaluations = self.problem.fe_count
            details = {"feasible": True, "fitness": fitness}
            return v, fitness, details

        solution = self._decode_and_repair(vector)
        violation = self.problem.constraints.check(solution)
        if not violation.feasible:
            raise RuntimeError(
                "Constraint repair returned an infeasible solution; refusing to "
                "evaluate it as a valid optimizer candidate."
            )
        fitness = float(self.problem.evaluate(solution, count_fe=True))
        self.evaluations = self.problem.fe_count
        details = {"feasible": True, "fitness": fitness}
        return solution, fitness, details

    def _decode_and_repair(self, vector: np.ndarray) -> StationSolution:
        """Map continuous coordinates to a feasible discrete solution efficiently."""
        v = np.asarray(vector, dtype=float).reshape(self.problem.k, 2)
        v = np.clip(v, self.lb, self.ub)

        candidates = self.problem.candidates.coordinates
        indices = self.problem.candidates.indices
        min_dist = self.problem.min_station_distance

        selected_coords = []
        selected_indices = []

        for station_no in range(self.problem.k):
            target = v[station_no]
            dist_sq = np.sum((candidates - target) ** 2, axis=1)
            candidate_order = np.argsort(dist_sq)

            for idx in candidate_order:
                candidate_idx = int(indices[idx])
                if candidate_idx in selected_indices:
                    continue
                coord = candidates[idx]

                if not selected_coords:
                    selected_coords.append(coord)
                    selected_indices.append(candidate_idx)
                    break

                dists = np.linalg.norm(np.asarray(selected_coords) - coord, axis=1)
                if np.all(dists >= min_dist):
                    selected_coords.append(coord)
                    selected_indices.append(int(indices[idx]))
                    break

        # Đảm bảo chuyển đổi nghiệm rời rạc và sửa nghiệm an toàn 100% (Giải pháp 1: Back-off Repair)
        if len(selected_indices) == self.problem.k:
            solution = StationSolution(np.asarray(selected_indices, dtype=int))
            if not self.problem.constraints.check(solution).feasible:
                solution = self.problem.repair(solution, self.rng)
        elif len(selected_indices) > 0:
            partial_solution = StationSolution(np.asarray(selected_indices, dtype=int))
            solution = self.problem.repair(partial_solution, self.rng)
        else:
            solution = self.problem.random_solution(self.rng)

        if not self.problem.constraints.check(solution).feasible:
            # Fallback tuyệt đối về feasible witness đã chứng thực nếu mọi nỗ lực quay lui đều thất bại
            solution = StationSolution(self.problem._feasible_witness.candidate_indices.copy())

        return solution

    def _random_vector(self) -> np.ndarray:
        return self.rng.uniform(self.lb_vec, self.ub_vec)

    def _initialize(self):
        """Initialize population. Each agent costs exactly 1 FE. No redundant details calls."""
        population = []

        if self.is_continuous:
            for _ in range(self.config.population_size):
                if self._budget_exhausted():
                    break
                vector = self._random_vector()
                fitness = float(self.problem.evaluate(vector, count_fe=True))
                self.evaluations = self.problem.fe_count
                population.append(
                    AgentState(
                        vector=vector,
                        solution=vector,
                        fitness=fitness,
                        feasible=True,
                    )
                )
            return population

        for _ in range(self.config.population_size):
            if self._budget_exhausted():
                break
            solution = self.problem.random_solution(self.rng)
            if not self.problem.constraints.check(solution).feasible:
                raise RuntimeError(
                    "random_solution returned an infeasible solution; "
                    "refusing to initialize the optimizer with it."
                )
            coords = solution.coordinates(self.problem.candidates)
            vector = coords.reshape(-1).copy()
            fitness = float(self.problem.evaluate(solution, count_fe=True))
            self.evaluations = self.problem.fe_count
            feasible = self.problem.constraints.check(solution).feasible

            # Avoid re-evaluating objective for details in initialization (costs 0 FE)
            population.append(
                AgentState(
                    vector=vector,
                    solution=solution,
                    fitness=fitness,
                    feasible=feasible,
                )
            )

        return population

    @staticmethod
    def _update_agent(agent, vector, solution, fitness, details):
        if fitness < agent.fitness:
            agent.vector = vector.copy()
            agent.solution = solution
            agent.fitness = float(fitness)
            agent.feasible = details["feasible"]
            agent.stagnation = 0
            return True

        agent.stagnation += 1
        return False

    def _best(self, population):
        return min(population, key=lambda a: a.fitness)

    def optimize(self) -> OptimizationResult:
        raise NotImplementedError
