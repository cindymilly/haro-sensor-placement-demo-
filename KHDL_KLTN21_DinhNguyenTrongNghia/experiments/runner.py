from __future__ import annotations

import os
import csv
import time
import argparse
from typing import Dict, List, Any
import numpy as np

from optimizers.base import OptimizerConfig
from optimizers.haro import HARO
from optimizers.aro import ARO
from optimizers.pso import PSO
from optimizers.mvo import MVO
from optimizers.ga import GA

from visualization.plots import (
    plot_heatmap_with_stations,
    plot_pollution_dashboard,
    plot_convergence_comparison,
)
from .scenario_builder import (
    build_experiment_problem,
    scaled_params,
    create_scaled_sources,
)

import concurrent.futures

ALGORITHMS = {
    "HARO": HARO,
    "ARO": ARO,
    "PSO": PSO,
    "MVO": MVO,
    "GA": GA,
}

DEFAULT_GRIDS = [(50, 50), (70, 70), (100, 100)]
DEFAULT_KS = [5, 10, 15]

# Thư mục lưu kết quả duy nhất theo đúng yêu cầu:
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ket_qua_thi_nghiem")


def _run_single_opt(args):
    algo_cls, problem, pop_size, T, seed, run_idx, total_pollution = args
    problem.repair_fallback_count = 0
    problem.random_solution_fallback_count = 0
    config = OptimizerConfig(
        population_size=pop_size,
        max_iterations=T,
        seed=seed,
        verbose=False,
    )
    opt = algo_cls(problem, config)
    t0 = time.perf_counter()
    res = opt.optimize()
    elapsed = time.perf_counter() - t0

    # Kiểm toán tính khả thi: Nếu có bất kỳ trạm nào vi phạm d_min, kích hoạt cơ chế sửa nghiệm an toàn
    if not problem.constraints.check(res.best_solution).feasible:
        print(
            f"    [CẢNH BÁO TỰ PHỤC HỒI] Run {run_idx+1}: {res.algorithm} sinh nghiệm vi phạm d_min. "
            "Kích hoạt cơ chế quay lui (Back-off Repair) để bảo đảm 100% khả thi..."
        )
        res.best_solution = problem.repair(res.best_solution, np.random.default_rng(seed))
        if not problem.constraints.check(res.best_solution).feasible:
            res.best_solution = StationSolution(problem._feasible_witness.candidate_indices.copy())
        
        # Cập nhật lại chi tiết fitness sau khi repair
        res.best_fitness = float(problem.evaluate(res.best_solution, count_fe=False))
        res.best_details = problem.evaluate_with_details(res.best_solution, count_fe=False)

    cov_val = res.best_details.get("covered_weight", res.best_details.get("coverage_value", 0.0))
    cov_pct = res.best_details.get("coverage_percent", 100.0 * cov_val / total_pollution)

    return {
        "run_idx": run_idx,
        "fitness": res.best_fitness,
        "coverage": cov_val,
        "cov_pct": cov_pct,
        "elapsed": elapsed,
        "convergence": res.convergence,
        "res": res,
        "repair_fallback_count": problem.repair_fallback_count,
        "random_solution_fallback_count": problem.random_solution_fallback_count,
    }


def run_scenario(
    M: int,
    N: int,
    K: int,
    algorithms: Dict[str, Any],
    output_dir: str,
    n_runs: int = 30,
    workers: int = 1,
    on_row_done=None,
) -> List[Dict[str, Any]]:
    """Chạy thực nghiệm n_runs lần độc lập cho tất cả các thuật toán trong kịch bản (M, N, K)."""
    problem, env_result, feasibility_mask, R_cover, d_min, pop_size, T = build_experiment_problem(M, N, K)
    total_pollution = float(env_result.concentration.sum())

    print(f"\n================================================================================")
    print(f"Kịch bản: Lưới {M}x{N} | K={K} trạm | pop_size={pop_size} | T={T} | R_cover={R_cover} | d_min={d_min}")
    print(f"Số lần chạy độc lập: {n_runs} | Thuật toán so sánh: {list(algorithms.keys())} | Workers: {workers}")
    print(f"================================================================================")

    tag = f"grid{M}x{N}_K{K}"
    algo_convergences: Dict[str, List[np.ndarray]] = {}
    scenario_rows = []

    best_haro_result = None
    best_haro_fitness = float("inf")

    for algo_name, algo_cls in algorithms.items():
        print(f"\n  --- Đang chạy [{algo_name}] ({n_runs} runs) ---")
        fitness_list, coverage_list, time_list, convergence_list = [], [], [], []
        repair_fallback_counts, random_fallback_counts = [], []

        tasks = [
            (algo_cls, problem, pop_size, T, 1000 * run_idx + 7, run_idx, total_pollution)
            for run_idx in range(n_runs)
        ]

        if workers > 1 and n_runs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                run_results = list(executor.map(_run_single_opt, tasks))
            run_results.sort(key=lambda x: x["run_idx"])
        else:
            run_results = [_run_single_opt(task) for task in tasks]

        for r in run_results:
            fitness_list.append(r["fitness"])
            coverage_list.append(r["coverage"])
            time_list.append(r["elapsed"])
            convergence_list.append(r["convergence"])
            repair_fallback_counts.append(r["repair_fallback_count"])
            random_fallback_counts.append(r["random_solution_fallback_count"])

            # Luôn lưu giữ cá thể tốt nhất của HARO để vẽ Heatmap
            if algo_name == "HARO" and r["fitness"] < best_haro_fitness:
                best_haro_fitness = r["fitness"]
                best_haro_result = r["res"]

            print(
                f"    [{algo_name} Run {r['run_idx']+1:2d}/{n_runs}] "
                f"Fitness: {r['fitness']:11.3f} | Cov: {r['coverage']:10.2f} "
                f"({r['cov_pct']:5.2f}%) | {r['elapsed']:5.2f}s"
            )

        fitness_arr = np.array(fitness_list)
        cov_pct_arr = 100.0 * np.array(coverage_list) / total_pollution
        algo_convergences[algo_name] = convergence_list

        row = {
            "grid": f"{M}x{N}",
            "K": K,
            "algorithm": algo_name,
            "pop_size": pop_size,
            "T": T,
            "R_cover": R_cover,
            "d_min": d_min,
            "n_runs": n_runs,
            "fitness_mean": round(float(fitness_arr.mean()), 4),
            "fitness_std": round(float(fitness_arr.std()), 4),
            "fitness_best": round(float(fitness_arr.min()), 4),
            "coverage_pct_mean": round(float(cov_pct_arr.mean()), 2),
            "coverage_pct_std": round(float(cov_pct_arr.std()), 2),
            "coverage_pct_best": round(float(cov_pct_arr.max()), 2),
            "time_s_per_run_mean": round(float(np.mean(time_list)), 2),
            "time_s_total": round(float(np.sum(time_list)), 2),
            "repair_fallback_count_total": int(sum(repair_fallback_counts)),
            "random_solution_fallback_count_total": int(sum(random_fallback_counts)),
        }
        scenario_rows.append(row)

        print(
            f"  => [{algo_name}] Mean Fitness: {fitness_arr.mean():.3f} (STD: {fitness_arr.std():.3f}) | "
            f"X_best Fitness: {fitness_arr.min():.3f} | Mean Cov: {cov_pct_arr.mean():.2f}%"
        )

        # 1. GHI CSV NGAY LẬP TỨC SAU MỖI THUẬT TOÁN (Real-time CSV update)
        if on_row_done is not None:
            on_row_done(row)

        # 2. XUẤT HEATMAP NGAY LẬP TỨC KHI HARO VỪA HOÀN TẤT (Không cần chờ các thuật toán khác)
        if algo_name == "HARO" and best_haro_result is not None:
            best_stations = best_haro_result.best_solution.coordinates(problem.candidates)
            best_cov_val = best_haro_result.best_details.get("covered_weight", best_haro_result.best_details.get("coverage_value", 0.0))
            best_cov_pct = best_haro_result.best_details.get("coverage_percent", 100.0 * best_cov_val / total_pollution)

            heatmap_path = os.path.join(output_dir, f"heatmap_{tag}.png")
            os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
            plot_heatmap_with_stations(
                pollution_grid=env_result.concentration,
                stations=best_stations,
                R_cover=R_cover,
                feasibility_mask=feasibility_mask,
                sources=env_result.sources,
                F=best_cov_val,
                coverage_pct=best_cov_pct,
                title=f"Heatmap + Trạm Quan Trắc HARO — Lưới {M}x{N}, K={K}",
                save_path=heatmap_path,
            )
            print(f"  [Heatmap CẬP NHẬT NGAY] Đã xuất bản đồ trạm tối ưu HARO: {heatmap_path}")

        # 3. CẬP NHẬT BIỂU ĐỒ HỘI TỤ NGAY
        conv_path = os.path.join(output_dir, f"convergence_{tag}.png")
        os.makedirs(os.path.dirname(conv_path), exist_ok=True)
        plot_convergence_comparison(
            algo_runs=algo_convergences,
            title=f"Đường hội tụ thuật toán — Lưới {M}x{N}, K={K} ({n_runs} runs)",
            save_path=conv_path,
        )

    return scenario_rows


def load_existing_results(csv_path: str, n_runs: int = 30) -> tuple[List[Dict[str, Any]], set[tuple[str, int]]]:
    """Đọc kết quả cũ từ CSV nếu có và xác định các kịch bản (grid, K) đã hoàn thành đủ số thuật toán VÀ đủ n_runs."""
    if not os.path.exists(csv_path):
        return [], set()

    rows = []
    completed_scenarios = set()
    scenario_counts: Dict[tuple[str, int], set[str]] = {}

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required_columns = {
                "repair_fallback_count_total",
                "random_solution_fallback_count_total",
            }
            if not required_columns.issubset(set(reader.fieldnames or [])):
                print(
                    "  [Cảnh báo] CSV hiện có chưa ghi bộ đếm repair fallback; "
                    "bỏ qua dữ liệu cũ để chạy lại theo schema mới."
                )
                return [], set()
            for row in reader:
                rows.append(row)
                grid = row.get("grid", "")
                k_str = row.get("K", "0")
                algo = row.get("algorithm", "")
                try:
                    k = int(k_str)
                    key = (grid, k)
                    row_runs = int(row.get("n_runs", 0))
                    if row_runs >= n_runs:
                        if key not in scenario_counts:
                            scenario_counts[key] = set()
                        scenario_counts[key].add(algo)
                except ValueError:
                    pass

        for key, algos in scenario_counts.items():
            if set(ALGORITHMS.keys()).issubset(algos):
                completed_scenarios.add(key)
    except Exception as e:
        print(f"  [Cảnh báo] Không thể đọc file CSV cũ ({e}), sẽ bắt đầu mới.")
        return [], set()

    return rows, completed_scenarios


def run_experiments(
    grids: List[tuple[int, int]] = DEFAULT_GRIDS,
    k_values: List[int] = DEFAULT_KS,
    n_runs: int = 30,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    algorithms: Dict[str, Any] = None,
    resume: bool = True,
    workers: int = 4,
):
    """Điều phối toàn bộ thực nghiệm đa kịch bản cho 5 thuật toán, lưu duy nhất vào output_dir."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if algorithms is None:
        algorithms = ALGORITHMS

    csv_path = os.path.join(output_dir, "bang_tong_hop_ket_qua.csv")
    all_rows = []
    completed_keys = set()

    if resume and os.path.exists(csv_path):
        all_rows, completed_keys = load_existing_results(csv_path, n_runs=n_runs)
        if completed_keys:
            print(f"\n[Resume] Đã phát hiện {len(completed_keys)} kịch bản đã hoàn thành trong {csv_path}:")
            for grid_str, k_val in sorted(completed_keys):
                print(f"   ✓ Lưới {grid_str}, K={k_val} (đã đủ 5 thuật toán)")

    # Sinh bản đồ tổng quan dashboard ô nhiễm 2 panel trên lưới lớn nhất
    max_grid = max(grids, key=lambda g: g[0] * g[1])
    dash_path = os.path.join(output_dir, "dashboard_pollution_overview.png")
    if not os.path.exists(dash_path):
        sources_max = create_scaled_sources(max_grid[0], max_grid[1])
        _, env_max, _, _, _, _, _ = build_experiment_problem(max_grid[0], max_grid[1], k_values[0])
        plot_pollution_dashboard(
            pollution_grid=env_max.concentration,
            sources=env_max.sources,
            title=f"Bản Đồ Ô Nhiễm Không Khí — Khu Đô Thị Giả Định ({max_grid[0]}x{max_grid[1]})",
            save_path=dash_path,
        )
        print(f"[Khởi tạo] Đã tạo dashboard tổng quan: {dash_path}")

    def on_row_done(new_row):
        """Ghi nhận ngay lập tức mỗi khi 1 thuật toán chạy xong vào CSV."""
        found = False
        for idx, r in enumerate(all_rows):
            if r.get("grid") == new_row.get("grid") and str(r.get("K")) == str(new_row.get("K")) and r.get("algorithm") == new_row.get("algorithm"):
                all_rows[idx] = new_row
                found = True
                break
        if not found:
            all_rows.append(new_row)

        fieldnames = list(all_rows[0].keys())
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"  [CSV CẬP NHẬT NGAY] Đã ghi dòng [{new_row['algorithm']}] vào CSV: {csv_path} ({len(all_rows)} dòng)")
        except Exception as err:
            print(f"  [Cảnh báo ghi CSV {csv_path}]: {err}")

    for (M, N) in grids:
        grid_key = f"{M}x{N}"
        for K in k_values:
            if resume and (grid_key, K) in completed_keys:
                print(f"\n>> [BỎ QUA - ĐÃ HOÀN TẤT] Lưới {grid_key}, K={K} đã có đầy đủ kết quả trong CSV.")
                continue

            rows = run_scenario(
                M=M,
                N=N,
                K=K,
                algorithms=algorithms,
                output_dir=output_dir,
                n_runs=n_runs,
                workers=workers,
                on_row_done=on_row_done,
            )

    print(f"\n================================================================================")
    print(f"ĐÃ HOÀN TẤT THỰC NGHIỆM TẤT CẢ CÁC KỊCH BẢN YÊU CẦU!")
    print(f"File bảng tổng hợp kết quả CSV: {csv_path}")
    print(f"Toàn bộ hình ảnh (heatmap, convergence 5 thuật toán, dashboard) tại: {output_dir}/")
    print(f"================================================================================")


def parse_grids_arg(grids_str: str) -> List[tuple[int, int]]:
    grids = []
    for item in grids_str.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            grids.append((int(parts[0]), int(parts[1])))
        else:
            raise argparse.ArgumentTypeError(f"Định dạng lưới không hợp lệ: '{item}'. Ví dụ hợp lệ: '50x50' hoặc '50x50,70x70'")
    return grids


def parse_k_arg(k_str: str) -> List[int]:
    ks = []
    for item in k_str.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            ks.append(int(item))
        else:
            raise argparse.ArgumentTypeError(f"Giá trị K không hợp lệ: '{item}'. Ví dụ: '15' hoặc '5,10,15'")
    return ks


def main():
    cpu_cores = os.cpu_count() or 4
    default_workers = max(1, min(cpu_cores - 1, 6))

    parser = argparse.ArgumentParser(description="Chạy thực nghiệm so sánh 5 thuật toán tối ưu mạng trạm quan trắc.")
    parser.add_argument("--runs", type=int, default=30, help="Số lần chạy độc lập mỗi kịch bản (mặc định: 30)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất ảnh và file CSV")
    parser.add_argument("--quick-test", action="store_true", help="Chạy kiểm thử nhanh (lưới 50x50, K=5, 2 runs) để xác nhận pipeline")
    parser.add_argument("--grids", type=str, default=None, help="Lưới cần chạy, cách nhau bằng dấu phẩy (vd: '50x50' hoặc '50x50,70x70')")
    parser.add_argument("--k-values", type=str, default=None, help="Các mức K cần chạy, cách nhau bằng dấu phẩy (vd: '15' hoặc '5,10,15')")
    parser.add_argument("--resume", action="store_true", default=True, help="Tự động tiếp tục từ kết quả CSV cũ nếu có (mặc định: BẬT)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Không resume, chạy lại từ đầu")
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Số tiến trình CPU chạy song song (mặc định: {default_workers})")
    args = parser.parse_args()

    if args.quick_test:
        quick_output_dir = (
            args.output_dir if args.output_dir != DEFAULT_OUTPUT_DIR
            else os.path.join(PROJECT_ROOT, "ket_qua_quick_test")
        )
        print(f"[Chế độ Quick-Test] Chạy nhanh lưới 50x50, K=5, 2 runs (lưu tại: {quick_output_dir})...")
        run_experiments(
            grids=[(50, 50)],
            k_values=[5],
            n_runs=2,
            output_dir=quick_output_dir,
            resume=False,
            workers=1,
        )
    else:
        grids = parse_grids_arg(args.grids) if args.grids else DEFAULT_GRIDS
        k_values = parse_k_arg(args.k_values) if args.k_values else DEFAULT_KS
        run_experiments(
            grids=grids,
            k_values=k_values,
            n_runs=args.runs,
            output_dir=args.output_dir,
            resume=args.resume,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()

