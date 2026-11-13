import numpy as np
import pytest

from .gaussian import GaussianDispersion
from .grid import Grid
from .scenario import create_scenario
from .types import EmissionSource, GridConfig, SourceType


def test_grid_sizes():
    for size in (50, 70, 100):
        grid = Grid(GridConfig(size, size))
        assert grid.shape == (size, size)
        assert grid.n_cells == size * size


def test_gaussian_peak_is_at_source():
    grid = Grid(GridConfig(21, 21))
    source = EmissionSource(
        x=10,
        y=10,
        source_type=SourceType.TRAFFIC,
        emission_rate=1.0,
    )

    model = GaussianDispersion(sigma=8.0)
    field = model.source_field(grid, source)

    row, col = grid.nearest_cell(10, 10)
    assert np.isclose(field[row, col], 1.0)
    assert np.isclose(field.max(), 1.0)


def test_gaussian_decreases_with_distance():
    grid = Grid(GridConfig(41, 41))
    source = EmissionSource(
        x=20,
        y=20,
        source_type=SourceType.TRAFFIC,
        emission_rate=1.0,
    )

    field = GaussianDispersion(sigma=8.0).source_field(grid, source)

    center = field[20, 20]
    near = field[20, 21]
    far = field[20, 30]

    assert center > near > far


def test_multiple_sources_are_additive():
    grid = Grid(GridConfig(21, 21))
    source_a = EmissionSource(5, 10, SourceType.TRAFFIC, 1.0)
    source_b = EmissionSource(15, 10, SourceType.INDUSTRIAL, 2.0)

    model = GaussianDispersion(sigma=8.0)
    combined = model.simulate(grid, [source_a, source_b])
    separate = (
        model.source_field(grid, source_a)
        + model.source_field(grid, source_b)
    )

    assert np.allclose(combined, separate)


def test_scenario_output_contract():
    result = create_scenario(
        grid_size=50,
        source_definitions=[
            {"x": 10, "y": 10, "source_type": "traffic"},
            {"x": 30, "y": 30, "source_type": "industrial"},
        ],
        sigma=8.0,
    )

    assert result.concentration.shape == (50, 50)
    assert result.grid_x.shape == (50, 50)
    assert result.grid_y.shape == (50, 50)
    assert len(result.sources) == 2
    assert result.metadata["sigma_default"] == 8.0


def test_invalid_sigma():
    with pytest.raises(ValueError):
        GaussianDispersion(sigma=0)
