from __future__ import annotations

import numpy as np

from environment.types import EnvironmentResult
from .candidates import CandidateSet
from .encoding import StationSolution


class CoverageObjective:
    """Pollution-weighted spatial coverage objective with continuous distance decay.

    Continuous spatial coverage formulation:
        Coverage(x, y, S) = max_{k=1..K} exp(-lambda * d((x, y), s_k))
        F_cov(S) = sum_{x, y} C(x, y) * Coverage(x, y, S)
        coverage_ratio(S) = F_cov(S) / sum_{x, y} C(x, y)

    The optimizer minimizes negative coverage ratio:
        fitness(S) = - coverage_ratio(S)

    Parameters:
        environment: Simulation environment containing grid coordinates and concentration field C(x, y).
        candidates: Discrete candidate set of potential station locations.
        coverage_radius: Effective coverage radius R_sens where coverage decays to 10% (exp(-lambda * R) = 0.1).
        decay_lambda: Optional explicit distance decay parameter lambda. If None, defaults to ln(10) / coverage_radius.
        normalize: Whether to normalize by total pollution mass into [0, 1].
    """

    def __init__(
        self,
        environment: EnvironmentResult,
        candidates: CandidateSet,
        coverage_radius: float = 8.0,
        decay_lambda: float | None = None,
        normalize: bool = True,
    ):
        if coverage_radius <= 0:
            raise ValueError("coverage_radius must be > 0.")

        concentration = np.asarray(environment.concentration, dtype=float)
        if concentration.ndim != 2:
            raise ValueError("environment.concentration must be a 2D matrix.")
        if np.any(~np.isfinite(concentration)):
            raise ValueError("Concentration contains non-finite values.")
        if np.any(concentration < 0):
            raise ValueError("Concentration must be non-negative.")

        self.environment = environment
        self.candidates = candidates
        self.coverage_radius = float(coverage_radius)
        self.normalize = bool(normalize)

        if decay_lambda is not None:
            if decay_lambda <= 0:
                raise ValueError("decay_lambda must be > 0.")
            self.decay_lambda = float(decay_lambda)
        else:
            # exp(-lambda * R) = 0.1 => lambda = ln(10) / R
            self.decay_lambda = float(np.log(10.0) / self.coverage_radius)

        self._grid_coordinates = np.column_stack(
            [environment.grid_x.ravel(), environment.grid_y.ravel()]
        )
        self._weights = concentration.ravel()

        total = float(np.sum(self._weights))
        self._total_weight = total

        # Precompute candidate -> cell distance decay factors for fast vectorized evaluation
        # decay_factors shape: (num_candidates, num_grid_cells)
        self._decay_factors = self._build_decay_factors()
        self._index_to_pos = {int(idx): i for i, idx in enumerate(self.candidates.indices)}

    def _build_decay_factors(self) -> np.ndarray:
        if self.candidates.size == 0 or len(self._grid_coordinates) == 0:
            return np.zeros((0, len(self._grid_coordinates)), dtype=float)

        dx = self._grid_coordinates[:, 0][None, :] - self.candidates.coordinates[:, 0][:, None]
        dy = self._grid_coordinates[:, 1][None, :] - self.candidates.coordinates[:, 1][:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        return np.exp(-self.decay_lambda * dist)

    def cell_coverage(self, solution: StationSolution) -> np.ndarray:
        """Return the continuous coverage factor in [0, 1] for every grid cell."""
        positions = [
            self._index_to_pos[int(idx)]
            for idx in solution.candidate_indices
            if int(idx) in self._index_to_pos
        ]

        if not positions:
            return np.zeros(len(self._grid_coordinates), dtype=float)

        # Max coverage across all active stations: max_{k=1..K} exp(-lambda * d)
        return np.max(self._decay_factors[positions], axis=0)

    def covered_mask(self, solution: StationSolution, threshold: float = 0.1) -> np.ndarray:
        """Boolean mask of cells with coverage factor >= threshold (for visualization compatibility)."""
        return self.cell_coverage(solution) >= threshold

    def coverage_ratio(self, solution: StationSolution) -> float:
        """Return the weighted coverage ratio in [0, 1]."""
        coverage_factors = self.cell_coverage(solution)
        covered_weight = float(np.dot(self._weights, coverage_factors))

        if self.normalize:
            if self._total_weight <= 0:
                return 0.0
            return float(covered_weight / self._total_weight)

        return covered_weight

    def fitness(self, solution: StationSolution) -> float:
        """Scalar objective to minimize."""
        return -float(self.coverage_ratio(solution))

    def details(self, solution: StationSolution) -> dict:
        cov_factors = self.cell_coverage(solution)
        covered_weight = float(np.dot(self._weights, cov_factors))
        ratio = float(covered_weight / self._total_weight) if self._total_weight > 0 else 0.0
        mask_eff = cov_factors >= 0.1

        return {
            "fitness": self.fitness(solution),
            "coverage_ratio": ratio,
            "coverage_percent": ratio * 100.0,
            "covered_cells": int(np.sum(mask_eff)),
            "total_cells": int(len(cov_factors)),
            "covered_weight": covered_weight,
            "total_weight": float(self._total_weight),
            "effective_coverage_cells": int(np.sum(mask_eff)),
            "mean_cell_coverage": float(np.mean(cov_factors)),
        }
