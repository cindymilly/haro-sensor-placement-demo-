from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np

from environment.scenario import create_scenario
from environment.grid import Grid
from environment.types import GridConfig, EnvironmentResult
from problem.candidates import CandidateSet
from problem.problem import MonitoringProblem

# Lưới cơ sở 60x60 thiết kế các nguồn thải chuẩn
_BASE_GRID = 60.0
_BASE_SOURCES = [
    # (x, y, emission_rate, sigma, source_type)
    # Giao thông
    (10, 10, 85.0, 5.0, "traffic"),
    (48, 9,  78.0, 5.0, "traffic"),
    (28, 42, 65.0, 4.0, "traffic"),
    # Công nghiệp
    (45, 42, 100.0, 7.0, "industrial"),
    (14, 46, 92.0,  6.0, "industrial"),
    (54, 26, 88.0,  6.0, "industrial"),
    # Xây dựng
    (6, 30,  55.0, 4.0, "construction"),
    (36, 14, 60.0, 4.0, "construction"),
    (52, 54, 58.0, 4.0, "construction"),
    # Nhiễu môi trường
    (30, 52, 70.0, 5.0, "noise"),
    (20, 20, 62.0, 5.0, "noise"),
    (44, 18, 66.0, 5.0, "noise"),
]


def create_scaled_sources(M: int, N: int) -> List[Dict[str, Any]]:
    """Sinh danh sách 12 nguồn thải co giãn theo tỷ lệ kích thước lưới M x N."""
    sx = M / _BASE_GRID
    sy = N / _BASE_GRID
    s_avg = (sx + sy) / 2.0

    sources = []
    for x, y, rate, sig, stype in _BASE_SOURCES:
        sources.append({
            "x": float(x * sx),
            "y": float(y * sy),
            "emission_rate": float(rate),
            "sigma": float(sig * s_avg),
            "source_type": stype,
        })
    return sources


def create_feasibility_mask(M: int, N: int) -> np.ndarray:
    """Tạo mặt nạ vùng cấm đặt trạm (1: hợp lệ, 0: vùng cấm).

    Được chuẩn hóa theo tỷ lệ % kích thước lưới:
      - Vùng cấm 1 (hồ / công viên trung tâm): [0.33M - 0.50M, 0.33N - 0.50N]
      - Vùng cấm 2 (khu quân sự / an ninh):    [0.08M - 0.20M, 0.58N - 0.70N]
    """
    mask = np.ones((M, N), dtype=int)

    def zone(x0_frac, x1_frac, y0_frac, y1_frac):
        x0, x1 = int(round(x0_frac * M)), int(round(x1_frac * M))
        y0, y1 = int(round(y0_frac * N)), int(round(y1_frac * N))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        mask[x0:x1, y0:y1] = 0

    zone(0.33, 0.50, 0.33, 0.50)
    zone(0.08, 0.20, 0.58, 0.70)
    return mask


def scaled_params(M: int, N: int, K: int, base_grid: int = 50) -> Tuple[int, int, float, float]:
    """Tự động co giãn tham số thuật toán & bài toán theo chuẩn bài báo khoa học.

    Chuẩn học thuật cho các thuật toán tối ưu hóa (ARO Wang 2022, PSO, MVO, GA):
      - Kích thước quần thể (pop_size): 30 cho lưới 50x50, co giãn nhẹ lên 35 (70x70) và 40 (100x100).
        (pop_size=30 là chuẩn vàng quốc tế trong hầu hết các bài báo Metaheuristics uy tín).
      - Số vòng lặp tối đa (T): 150 (lưới 50x50), 180 (70x70) và 200 (100x100).
        (Các thuật toán đều đã hội tụ hoàn toàn từ trước thế hệ 70-80).
      - R_cover, d_min: co giãn tuyến tính theo kích thước thực địa của đô thị.
    """
    area_scale = (M * N) / (base_grid * base_grid)
    side_scale = M / base_grid

    pop_size = int(round(30 + 5 * (area_scale - 1)))
    pop_size = max(25, min(45, pop_size))

    T = int(round(150 + 20 * (area_scale - 1)))
    T = max(150, min(220, T))

    R_cover = round(8.0 * side_scale, 2)
    d_min = round(10.0 * side_scale, 2)

    return pop_size, T, R_cover, d_min


def build_experiment_problem(
    M: int,
    N: int,
    K: int,
    random_seed: int = 42,
) -> Tuple[MonitoringProblem, EnvironmentResult, np.ndarray, float, float, int, int]:
    """Xây dựng bài toán tối ưu hoàn chỉnh cho kịch bản (M, N, K).

    Trả về:
      problem          : Đối tượng MonitoringProblem
      env_result       : Kết quả nồng độ ô nhiễm môi trường EnvironmentResult
      feasibility_mask : Mặt nạ vùng cấm (M, N)
      R_cover          : Bán kính phủ
      d_min            : Khoảng cách trạm tối thiểu
      pop_size         : Kích thước quần thể co giãn
      T                : Số vòng lặp tối đa co giãn
    """
    sources_defs = create_scaled_sources(M, N)
    feasibility_mask = create_feasibility_mask(M, N)

    env_result = create_scenario(
        grid_size=(M, N),
        sources=sources_defs,
        sigma=8.0,
        background_noise_std=0.5,
        random_seed=random_seed,
    )

    grid = Grid(GridConfig(width=M, height=N))
    candidates = CandidateSet.from_grid(grid, feasibility_mask=feasibility_mask)

    pop_size, T, R_cover, d_min = scaled_params(M, N, K)

    problem = MonitoringProblem(
        environment=env_result,
        candidates=candidates,
        k=K,
        coverage_radius=R_cover,
        min_station_distance=d_min,
        penalty=10.0,
    )

    return problem, env_result, feasibility_mask, R_cover, d_min, pop_size, T
