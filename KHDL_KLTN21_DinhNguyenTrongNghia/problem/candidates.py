from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from environment.grid import Grid


@dataclass(frozen=True)
class CandidateSet:
    """Discrete candidate locations for monitoring stations.

    Each candidate is a valid grid-cell center. The optimization algorithms
    operate on candidate indices instead of arbitrary continuous coordinates.
    """

    indices: np.ndarray       # shape (M,), integer flattened grid indices
    coordinates: np.ndarray   # shape (M, 2), columns are x, y

    def __post_init__(self):
        indices = np.asarray(self.indices)
        coordinates = np.asarray(self.coordinates, dtype=float)

        if indices.ndim != 1:
            raise ValueError("indices must be a 1D array.")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (M, 2).")
        if len(indices) != len(coordinates):
            raise ValueError("indices and coordinates must have the same length.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Candidate indices must be unique.")

        object.__setattr__(self, "indices", indices.astype(int, copy=False))
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "_index_to_pos", {int(idx): i for i, idx in enumerate(indices)})
        object.__setattr__(self, "_index_set", set(int(idx) for idx in indices))

    @classmethod
    def from_grid(cls, grid: Grid, feasibility_mask: np.ndarray | None = None) -> "CandidateSet":
        """Use every grid cell center as a candidate location, optionally excluding restricted zones."""
        coords = grid.flatten_coordinates()
        indices = np.arange(grid.n_cells, dtype=int)
        if feasibility_mask is not None:
            valid_mask = (feasibility_mask.flatten() > 0)
            indices = indices[valid_mask]
            coords = coords[valid_mask]
        return cls(indices=indices, coordinates=coords)

    @property
    def size(self) -> int:
        return len(self.indices)

    def coordinate_of(self, candidate_index: int) -> np.ndarray:
        pos = self._index_to_pos.get(int(candidate_index))
        if pos is None:
            raise KeyError(f"Unknown candidate index: {candidate_index}")
        return self.coordinates[pos]
