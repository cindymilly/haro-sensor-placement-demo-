"""Unit tests for HCMC official data and problem integration."""

import pytest
import numpy as np

from environment.hcmc_official_data import (
    HCMC_BOUNDS,
    HCMC_EXISTING_STATIONS,
    HCMC_REAL_HOTSPOTS,
    geo_to_grid,
    grid_to_geo,
    create_hcmc_feasibility_mask,
)
from problem.hcmc_real_problem import (
    build_hcmc_real_problem,
    get_hcmc_existing_network_solution,
)


def test_hcmc_official_stations_and_hotspots_count():
    assert len(HCMC_EXISTING_STATIONS) == 10
    assert len(HCMC_REAL_HOTSPOTS) == 12


def test_geo_coordinate_conversion():
    m, n = 70, 70
    lat, lon = HCMC_BOUNDS["lat_min"], HCMC_BOUNDS["lon_min"]
    x, y = geo_to_grid(lat, lon, m, n)
    assert x == 0 and y == 0

    rev_lat, rev_lon = grid_to_geo(0, 0, m, n)
    assert pytest.approx(rev_lat, abs=1e-3) == lat
    assert pytest.approx(rev_lon, abs=1e-3) == lon


def test_hcmc_feasibility_mask():
    mask = create_hcmc_feasibility_mask(50, 50)
    assert mask.shape == (50, 50)
    assert mask.sum() > 0
    assert (mask == 0).sum() > 0  # Waterway / airport excluded


def test_hcmc_real_problem_building_and_evaluation():
    problem, env_result, mask, Rc, dmin = build_hcmc_real_problem(grid_m=40, grid_n=40, K=5)
    assert problem.k == 5
    assert env_result.concentration.shape == (40, 40)
    assert Rc > 0
    assert dmin > 0

    sol, details = get_hcmc_existing_network_solution(problem, grid_m=40, grid_n=40)
    assert "coverage_percent" in details
    assert "covered_hotspots" in details
    assert details["covered_hotspots"] >= 0
