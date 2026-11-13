from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np

from .candidates import CandidateSet
from .encoding import StationSolution


@dataclass(frozen=True)
class ConstraintViolation:
    feasible: bool
    duplicate_count: int
    out_of_candidate_count: int
    min_pairwise_distance: float
    distance_violation: float
    quadratic_distance_violation: float = 0.0

    @property
    def total_violation(self) -> float:
        """Normalized total violation including quadratic distance violation."""
        return float(
            self.duplicate_count
            + self.out_of_candidate_count
            + self.distance_violation
            + self.quadratic_distance_violation
        )


class ConstraintChecker:
    """Checks the problem constraints without knowing any optimizer.
    
    Penalty formulation:
        Penalty(X) = gamma * sum_{i < j} max(0, d_min - d(s_i, s_j))^2
    """

    def __init__(
        self,
        candidates: CandidateSet,
        k: int,
        min_station_distance: float = 0.0,
    ):
        if k < 1:
            raise ValueError("k must be >= 1.")
        if k > candidates.size:
            raise ValueError("k cannot exceed the number of candidates.")
        if min_station_distance < 0:
            raise ValueError("min_station_distance must be >= 0.")

        self.candidates = candidates
        self.k = int(k)
        self.min_station_distance = float(min_station_distance)

    def check(self, solution: StationSolution) -> ConstraintViolation:
        ids = np.asarray(solution.candidate_indices, dtype=int)
        duplicate_count = len(ids) - len(np.unique(ids))

        valid_coords = []
        out_of_candidate_count = 0
        for idx in ids:
            pos = self.candidates._index_to_pos.get(int(idx))
            if pos is not None:
                valid_coords.append(self.candidates.coordinates[pos])
            else:
                out_of_candidate_count += 1

        quadratic_violation = 0.0
        if len(valid_coords) >= 2:
            coords = np.asarray(valid_coords, dtype=float)
            diff = coords[:, None, :] - coords[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            np.fill_diagonal(d2, np.inf)
            min_distance = float(np.sqrt(np.min(d2)))

            # Quadratic penalty across all pairs: sum_{i < j} max(0, d_min - d_ij)^2
            if self.min_station_distance > 0:
                dist_matrix = np.sqrt(d2)
                # Upper triangle indices
                triu_indices = np.triu_indices(len(valid_coords), k=1)
                pair_dists = dist_matrix[triu_indices]
                pair_viol = np.maximum(0.0, self.min_station_distance - pair_dists)
                quadratic_violation = float(np.sum(pair_viol ** 2))
        else:
            min_distance = float("inf")

        if self.min_station_distance == 0:
            distance_violation = 0.0
        elif np.isfinite(min_distance):
            distance_violation = max(
                0.0,
                (self.min_station_distance - min_distance)
                / self.min_station_distance,
            )
        else:
            distance_violation = 0.0

        feasible = (
            solution.k == self.k
            and duplicate_count == 0
            and out_of_candidate_count == 0
            and distance_violation <= 0.0
        )

        return ConstraintViolation(
            feasible=feasible,
            duplicate_count=duplicate_count,
            out_of_candidate_count=out_of_candidate_count,
            min_pairwise_distance=min_distance,
            distance_violation=distance_violation,
            quadratic_distance_violation=quadratic_violation,
        )
