from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from environment.types import EnvironmentResult
from .candidates import CandidateSet
from .constraints import ConstraintChecker, ConstraintViolation
from .encoding import StationSolution
from .objective import CoverageObjective


class BudgetExhaustedException(Exception):
    """Raised when the maximum allowable Function Evaluations (max_fes) is reached."""
    pass


class FeasibleSolutionSearchError(RuntimeError):
    """Raised when a feasible K-station witness could not be constructed.

    This is not proof that the instance is infeasible: the bounded randomized
    construction may have failed to find an existing feasible configuration.
    """

    pass


@dataclass
class MonitoringProblem:
    """Complete optimization-problem interface for Block 03 optimizers.

    Block 02 owns:
      - candidate locations,
      - K,
      - minimum station distance,
      - coverage radius,
      - objective/fitness,
      - feasibility checking,
      - centralized FE counter and budget enforcement.

    Block 02 does NOT own ARO/HARO/PSO/GA/MVO.
    """

    environment: EnvironmentResult
    candidates: CandidateSet
    k: int
    coverage_radius: float
    min_station_distance: float = 0.0
    penalty: float = 10.0
    max_fes: Optional[int] = None

    _fe_count: int = field(default=0, init=False)
    _fe_history: List[Tuple[int, float]] = field(default_factory=list, init=False)
    _feasible_witness: Optional[StationSolution] = field(default=None, init=False)
    repair_fallback_count: int = field(default=0, init=False)
    random_solution_fallback_count: int = field(default=0, init=False)

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k must be >= 1.")
        if self.k > self.candidates.size:
            raise ValueError("k cannot exceed candidate count.")
        if self.coverage_radius <= 0:
            raise ValueError("coverage_radius must be > 0.")
        if self.min_station_distance < 0:
            raise ValueError("min_station_distance must be >= 0.")
        if self.penalty < 0:
            raise ValueError("penalty must be >= 0.")

        self.objective = CoverageObjective(
            self.environment,
            self.candidates,
            coverage_radius=self.coverage_radius,
        )
        self.constraints = ConstraintChecker(
            self.candidates,
            k=self.k,
            min_station_distance=self.min_station_distance,
        )
        self._fe_count = 0
        self._fe_history = []
        self._feasible_witness = self._construct_feasible_solution(
            np.random.default_rng(0),
            initial_indices=(),
            max_restarts=128,
        )
        if self._feasible_witness is None:
            raise FeasibleSolutionSearchError(
                "Could not construct a feasible K-station configuration after 128 "
                "randomized attempts. The instance may be infeasible, or the bounded "
                "search may have missed a feasible configuration; this is not a proof "
                "of infeasibility. Check K, d_min, and the candidate mask."
            )

    def reset_fe(self, max_fes: Optional[int] = None) -> None:
        """Reset the centralized FE counter and optionally update budget limit."""
        self._fe_count = 0
        self._fe_history.clear()
        if max_fes is not None:
            self.max_fes = max_fes

    @property
    def fe_count(self) -> int:
        """Return the actual count of objective function evaluations."""
        return self._fe_count

    @property
    def fe_history(self) -> List[Tuple[int, float]]:
        """Return the sequence of (FE, fitness) logged during optimization."""
        return list(self._fe_history)

    def evaluate(self, solution: StationSolution, count_fe: bool = True) -> float:
        """Centralized evaluation entrypoint.
        
        Evaluates the objective fitness and penalizes constraint violations.
        Lower is better (canonical minimization).
        """
        if count_fe:
            if self.max_fes is not None and self._fe_count >= self.max_fes:
                raise BudgetExhaustedException(
                    f"Computational budget reached: FE = {self._fe_count} >= max_fes = {self.max_fes}"
                )
            self._fe_count += 1

        violation = self.constraints.check(solution)
        base_fitness = self.objective.fitness(solution)

        if violation.feasible:
            fit = float(base_fitness)
        else:
            fit = float(base_fitness + self.penalty * violation.total_violation)

        if count_fe:
            self._fe_history.append((self._fe_count, fit))

        return fit

    def evaluate_with_details(self, solution: StationSolution, count_fe: bool = False) -> dict:
        """Compute comprehensive metrics for reporting.
        
        By default, count_fe=False because reading back details of an existing solution
        does not constitute a new candidate evaluation (0 FE).
        """
        violation = self.constraints.check(solution)
        objective_details = self.objective.details(solution)

        base_fitness = float(objective_details["fitness"])
        if violation.feasible:
            penalized_fitness = base_fitness
        else:
            penalized_fitness = base_fitness + self.penalty * violation.total_violation

        if count_fe:
            if self.max_fes is not None and self._fe_count >= self.max_fes:
                raise BudgetExhaustedException(
                    f"Computational budget reached: FE = {self._fe_count} >= max_fes = {self.max_fes}"
                )
            self._fe_count += 1
            self._fe_history.append((self._fe_count, penalized_fitness))

        return {
            **objective_details,
            "feasible": violation.feasible,
            "duplicate_count": violation.duplicate_count,
            "out_of_candidate_count": violation.out_of_candidate_count,
            "min_pairwise_distance": violation.min_pairwise_distance,
            "distance_violation": violation.distance_violation,
            "quadratic_distance_violation": violation.quadratic_distance_violation,
            "total_violation": violation.total_violation,
            "penalized_fitness": penalized_fitness,
        }

    def random_solution(self, rng: np.random.Generator) -> StationSolution:
        """Return a feasible random solution, or a known feasible witness.

        If randomized construction reaches a dead end, it never inserts a
        closest-but-violating point. It falls back to the feasible witness found
        during problem initialization and records that fallback.
        """
        solution = self._construct_feasible_solution(
            rng, initial_indices=(), max_restarts=8
        )
        if solution is None:
            self.random_solution_fallback_count += 1
            solution = self._feasible_witness
        return StationSolution(solution.candidate_indices.copy())

    def _construct_feasible_solution(
        self,
        rng: np.random.Generator,
        initial_indices,
        max_restarts: int,
    ) -> Optional[StationSolution]:
        """Try randomized greedy completions; return None instead of violating d_min."""
        initial = [int(idx) for idx in initial_indices]
        if len(initial) > self.k or len(set(initial)) != len(initial):
            return None

        initial_positions = []
        for idx in initial:
            pos = self.candidates._index_to_pos.get(idx)
            if pos is None:
                return None
            initial_positions.append(pos)

        if len(initial_positions) > 1:
            initial_coords = self.candidates.coordinates[initial_positions]
            pair_delta = initial_coords[:, None, :] - initial_coords[None, :, :]
            pair_distances = np.sqrt(np.sum(pair_delta * pair_delta, axis=2))
            np.fill_diagonal(pair_distances, np.inf)
            if np.any(pair_distances < self.min_station_distance):
                return None

        for _ in range(max_restarts):
            selected_indices = list(initial)
            selected_positions = list(initial_positions)

            while len(selected_indices) < self.k:
                if not selected_positions:
                    eligible = np.arange(self.candidates.size)
                else:
                    selected_coords = self.candidates.coordinates[selected_positions]
                    distances = np.min(
                        np.linalg.norm(
                            self.candidates.coordinates[:, None, :]
                            - selected_coords[None, :, :],
                            axis=2,
                        ),
                        axis=1,
                    )
                    distances[selected_positions] = -np.inf
                    eligible = np.flatnonzero(
                        distances >= self.min_station_distance
                    )

                if len(eligible) == 0:
                    break

                chosen_pos = int(rng.choice(eligible))
                selected_positions.append(chosen_pos)
                selected_indices.append(int(self.candidates.indices[chosen_pos]))

            if len(selected_indices) == self.k:
                solution = StationSolution(np.asarray(selected_indices, dtype=int))
                if self.constraints.check(solution).feasible:
                    return solution

        return None

    def repair(self, solution: StationSolution,
               rng: np.random.Generator) -> StationSolution:
        """Cơ chế sửa nghiệm không gian kết hợp quay lui (Backtracking Spatial Repair).
        
        Quy trình chuẩn:
        1. Duyệt tuần tự để giữ lại tiền tố dài nhất các trạm đã thỏa mãn d_min.
        2. Thử hoàn tất ngẫu nhiên với tập an toàn C_safe = {c | min d(c, s) >= d_min}.
        3. Nếu gặp ngõ cụt (C_safe = empty): Kích hoạt quay lui (back-off), lần lượt rút ngắn
           tiền tố đã chọn để giải phóng không gian an toàn và thử hoàn tất lại (tối đa 4 lần).
        4. Nếu mọi nhánh quay lui đều bế tắc: Fallback về nghiệm chứng thực khả thi
           (feasible witness) đã được kiểm định từ lúc khởi tạo bài toán.
        => Bảo đảm 100% nghiệm đầu ra khả thi, tuyệt đối không bao giờ chọn điểm vi phạm d_min.
        """
        valid_indices = []
        valid_indices_set = set()
        valid_coords = []

        for idx in solution.candidate_indices:
            idx_int = int(idx)
            if idx_int in self.candidates._index_set and idx_int not in valid_indices_set:
                coord = self.candidates.coordinates[self.candidates._index_to_pos[idx_int]]
                if not valid_coords:
                    valid_indices.append(idx_int)
                    valid_indices_set.add(idx_int)
                    valid_coords.append(coord)
                else:
                    d = np.linalg.norm(np.asarray(valid_coords) - coord, axis=1)
                    if np.all(d >= self.min_station_distance):
                        valid_indices.append(idx_int)
                        valid_indices_set.add(idx_int)
                        valid_coords.append(coord)

        # Retry with progressively shorter accepted prefixes so an early greedy
        # choice cannot permanently block a feasible completion.
        for prefix_length in range(min(len(valid_indices), self.k), -1, -1):
            completed = self._construct_feasible_solution(
                rng,
                initial_indices=valid_indices[:prefix_length],
                max_restarts=4,
            )
            if completed is not None:
                return completed

        # The witness is known to satisfy every hard constraint. It may be less
        # aligned with the proposal, so expose fallback usage for auditability.
        self.repair_fallback_count += 1
        return StationSolution(self._feasible_witness.candidate_indices.copy())

    def summary(self) -> dict:
        return {
            "grid_shape": tuple(self.environment.concentration.shape),
            "candidate_count": self.candidates.size,
            "K": self.k,
            "coverage_radius": self.coverage_radius,
            "min_station_distance": self.min_station_distance,
            "objective": "maximize pollution-weighted continuous distance-decay coverage",
            "fitness_direction": "minimize",
            "actual_fe": self._fe_count,
            "max_fes": self.max_fes,
            "repair_fallback_count": self.repair_fallback_count,
            "random_solution_fallback_count": self.random_solution_fallback_count,
        }
