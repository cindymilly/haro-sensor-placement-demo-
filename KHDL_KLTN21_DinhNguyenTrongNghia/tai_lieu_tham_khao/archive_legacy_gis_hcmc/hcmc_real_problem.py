"""HCMC Real-World Environmental Optimization Problem Builder.

Tích hợp dữ liệu thực tế:
- Nguồn dữ liệu nồng độ: EU Copernicus CAMS PM2.5 Surface (Reanalysis data).
- Mặt nạ cấm trắc địa thực tế: Hệ thống Sông Sài Gòn, Sông Đồng Nai, vành đai an toàn hàng không TSN.
- 10 Trạm quan trắc thực tế hiện hữu (Baseline): HealthyAir (Mendeley Data DOI: 10.17632/pk6tzrjks8.1), US Consulate, DONRE HCMC.
- 12 Điểm nóng ô nhiễm thực tế: Cát Lái, Hiệp Phước, Tân Bình, An Sương, Xa lộ Hà Nội, SHTP...
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import numpy as np

from environment.types import EnvironmentResult, EmissionSource, GridConfig
from environment.grid import Grid
from environment.hcmc_official_data import (
    HCMC_BOUNDS,
    HCMC_EXISTING_STATIONS,
    HCMC_REAL_HOTSPOTS,
    geo_to_grid,
    grid_to_geo,
    create_hcmc_feasibility_mask,
    fetch_or_load_copernicus_surface,
)
from problem.candidates import CandidateSet
from problem.encoding import StationSolution
from problem.problem import MonitoringProblem


def build_hcmc_real_problem(
    grid_m: int = 70,
    grid_n: int = 70,
    K: int = 10,
    coverage_radius_km: float = 3.5,
    min_distance_km: float = 3.0,
    penalty: float = 10.0,
) -> Tuple[MonitoringProblem, EnvironmentResult, np.ndarray, float, float]:
    """Xây dựng bài toán tối ưu trạm quan trắc trên địa bàn thực địa TP.HCM.

    Tham số:
      grid_m, grid_n: Kích thước lưới phân giải (70x70 tương đương ô lưới ~ 350m x 400m).
      K: Số lượng trạm quan trắc cần triển khai (mặc định 10 trạm tương đương quy mô hiện hữu).
      coverage_radius_km: Bán kính cảm biến hiệu dụng (km).
      min_distance_km: Khoảng cách tối thiểu giữa 2 trạm tránh chồng lấn lãng phí (km).

    Trả về:
      problem: Đối tượng MonitoringProblem
      env_result: Kết quả nồng độ PM2.5 thực nghiệm
      feasibility_mask: Mặt nạ vùng cấm trắc địa
      R_cells: Bán kính cảm biến quy đổi sang ô lưới
      d_min_cells: Khoảng cách tối thiểu quy đổi sang ô lưới
    """
    # 1. Tải trường nồng độ PM2.5 thực tế từ Copernicus CAMS
    concentration = fetch_or_load_copernicus_surface(grid_m=grid_m, grid_n=grid_n)

    # 2. Xây dựng tọa độ lưới x, y
    gx, gy = np.meshgrid(np.arange(grid_m), np.arange(grid_n), indexing="ij")

    # 3. Chuyển đổi các điểm nóng ô nhiễm thực tế thành EmissionSource
    sources = []
    for spot in HCMC_REAL_HOTSPOTS:
        hx, hy = geo_to_grid(spot["lat"], spot["lon"], grid_m, grid_n)
        sig_cells = (spot["sigma_km"] / 25.0) * grid_m
        sources.append(
            EmissionSource(
                x=float(hx),
                y=float(hy),
                emission_rate=spot["emission_rate"],
                sigma=sig_cells,
                name=spot["name"],
                source_type=spot["type"],
            )
        )

    env_result = EnvironmentResult(
        concentration=concentration,
        grid_x=gx,
        grid_y=gy,
        sources=sources,
        metadata={
            "city": "Ho Chi Minh City, Vietnam",
            "source": "Copernicus Atmosphere Monitoring Service (CAMS) & Real Hotspots",
            "bounds": HCMC_BOUNDS,
            "grid_resolution_km": round(25.0 / grid_m, 3),
        },
    )

    # 4. Mặt nạ cấm trắc địa thực tế TP.HCM
    feasibility_mask = create_hcmc_feasibility_mask(grid_m, grid_n)

    # 5. Khởi tạo CandidateSet
    grid = Grid(GridConfig(width=grid_m, height=grid_n))
    candidates = CandidateSet.from_grid(grid, feasibility_mask=feasibility_mask)

    # 6. Quy đổi bán kính thực tế (km) sang ô lưới (cells)
    km_per_cell_x = 29.5 / grid_m
    km_per_cell_y = 23.3 / grid_n
    km_per_cell = (km_per_cell_x + km_per_cell_y) / 2.0

    R_cells = round(coverage_radius_km / km_per_cell, 2)
    d_min_cells = round(min_distance_km / km_per_cell, 2)

    problem = MonitoringProblem(
        environment=env_result,
        candidates=candidates,
        k=K,
        coverage_radius=R_cells,
        min_station_distance=d_min_cells,
        penalty=penalty,
    )

    return problem, env_result, feasibility_mask, R_cells, d_min_cells


def get_hcmc_existing_network_solution(
    problem: MonitoringProblem,
    grid_m: int = 70,
    grid_n: int = 70,
) -> Tuple[StationSolution, Dict[str, Any]]:
    """Chiếu mạng lưới 10 trạm quan trắc thực tế hiện hữu của TP.HCM vào không gian nghiệm.

    Dữ liệu từ HealthyAir Project (Mendeley Data), US Consulate, và Sở TN&MT TP.HCM.
    """
    selected_candidate_indices = []
    station_locations = []

    for sta in HCMC_EXISTING_STATIONS:
        gx, gy = geo_to_grid(sta["lat"], sta["lon"], grid_m, grid_n)
        # Tìm vị trí candidate gần nhất
        coords = problem.candidates.coordinates
        dist2 = (coords[:, 0] - gx) ** 2 + (coords[:, 1] - gy) ** 2
        best_cand_pos = int(np.argmin(dist2))
        cand_idx = int(problem.candidates.indices[best_cand_pos])
        cand_coord = coords[best_cand_pos]

        selected_candidate_indices.append(cand_idx)
        station_locations.append({
            "id": sta["id"],
            "name": sta["name"],
            "lat": sta["lat"],
            "lon": sta["lon"],
            "grid_x": int(cand_coord[0]),
            "grid_y": int(cand_coord[1]),
        })

    # Giữ đúng K trạm đầu tiên nếu danh sách dài hơn
    selected_candidate_indices = selected_candidate_indices[:problem.k]
    solution = StationSolution(np.array(selected_candidate_indices, dtype=int))

    eval_details = problem.evaluate_with_details(solution)

    # Đếm số điểm nóng thực tế được bao phủ
    covered_hotspots = 0
    hotspot_details = []
    for spot in HCMC_REAL_HOTSPOTS:
        hx, hy = geo_to_grid(spot["lat"], spot["lon"], grid_m, grid_n)
        # Tính khoảng cách tới trạm gần nhất
        is_covered = False
        for s in station_locations[:problem.k]:
            dist = np.hypot(s["grid_x"] - hx, s["grid_y"] - hy)
            if dist <= problem.coverage_radius:
                is_covered = True
                break
        if is_covered:
            covered_hotspots += 1
        hotspot_details.append({"spot": spot["name"], "covered": is_covered})

    return solution, {
        **eval_details,
        "covered_hotspots": covered_hotspots,
        "total_hotspots": len(HCMC_REAL_HOTSPOTS),
        "hotspot_coverage_rate": round(covered_hotspots / len(HCMC_REAL_HOTSPOTS) * 100, 2),
        "station_locations": station_locations[:problem.k],
    }
