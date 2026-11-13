from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .candidates import CandidateSet


@dataclass(frozen=True)
class StationSolution:
    """A decoded monitoring-network solution.

    candidate_indices contains exactly K distinct candidate IDs.
    """

    candidate_indices: np.ndarray

    def __post_init__(self):
        arr = np.asarray(self.candidate_indices, dtype=int)
        if arr.ndim != 1:
            raise ValueError("candidate_indices must be a 1D array.")
        if len(arr) == 0:
            raise ValueError("A solution must contain at least one station.")
        if len(np.unique(arr)) != len(arr):
            raise ValueError("A solution cannot contain duplicate station locations.")
        object.__setattr__(self, "candidate_indices", arr)

    @property
    def k(self) -> int:
        return len(self.candidate_indices)

    def coordinates(self, candidates: CandidateSet) -> np.ndarray:
        positions = []
        for idx in self.candidate_indices:
            pos = np.where(candidates.indices == idx)[0]
            if len(pos) == 0:
                raise KeyError(f"Unknown candidate index: {idx}")
            positions.append(candidates.coordinates[pos[0]])
        return np.asarray(positions, dtype=float)

    @classmethod
    def random(cls, candidates: CandidateSet, k: int, rng: np.random.Generator):
        if k < 1:
            raise ValueError("k must be >= 1.")
        if k > candidates.size:
            raise ValueError("k cannot exceed the number of candidates.")
        chosen = rng.choice(candidates.indices, size=k, replace=False)
        return cls(np.sort(chosen))
