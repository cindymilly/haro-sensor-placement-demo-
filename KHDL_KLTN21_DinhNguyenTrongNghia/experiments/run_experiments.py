"""Thực nghiệm chuẩn hóa Benchmark:
- 5 thuật toán: HARO (đề xuất), ARO, PSO, MVO, GA
- N = 30 runs độc lập, seed = 1..30
- Ngân sách đánh giá hàm mục tiêu: N_pop = 30, T = 100 iterations (FE = 3.000)
- Lưu toàn bộ kết quả thô: raw_runs_results.csv, convergence_traces.csv
- Phân tích thống kê: Mean, Std, Median, IQR, Kruskal-Wallis, Friedman, Mann-Whitney U, Wilcoxon, Holm, A12, Cliff's delta
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.stats as stats
import concurrent.futures
import matplotlib.pyplot as plt

# Ensure root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimizers.base import OptimizerConfig
from optimizers.haro import HARO
from optimizers.aro import ARO
from optimizers.pso import PSO
from optimizers.mvo import MVO
from optimizers.ga import GA

from experiments.scenario_builder import build_experiment_problem

ALGORITHMS = {
    "HARO": HARO,
    "ARO": ARO,
    "PSO": PSO,
    "MVO": MVO,
    "GA": GA,
}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ket_qua_thi_nghiem")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_a12_and_cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Calculate Vargha-Delaney A12 effect size and Cliff's delta.
    For minimization (lower fitness is better), group1 is better if fitness is lower.
    So we compare -group1 vs -group2 or count (x1 < x2).
    """
    n1 = len(group1)
    n2 = len(group2)
    # Higher is better for -fitness (or coverage ratio)
    # Let's count how often group1 is better than group2 (i.e. group1 < group2 in minimization)
    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    # Mann-Whitney U in scipy is sum of ranks - n1*(n1+1)/2 for group 1
    # For minimization: group1 having lower values means U for (-group1) vs (-group2)
    # Standard Vargha-Delaney A12 where group 1 wins if group1 < group2:
    wins = np.sum(group1[:, None] < group2[None, :])
    ties = np.sum(group1[:, None] == group2[None, :])
    a12 = (wins + 0.5 * ties) / (n1 * n2)
    cliffs_delta = 2.0 * a12 - 1.0
    return float(a12), float(cliffs_delta)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Apply Holm-Bonferroni step-down procedure."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_p = [0.0] * m
    significant = [False] * m

    running_max = 0.0
    for rank, (original_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        p_adj = min(1.0, p_val * multiplier)
        p_adj = max(p_adj, running_max)  # enforce monotonicity
        running_max = p_adj
        adjusted_p[original_idx] = float(p_adj)
        significant[original_idx] = bool(p_adj < alpha)

    return adjusted_p, significant


def run_single_trial(args: Tuple) -> Dict[str, Any]:
    """Run one optimization run for a given instance, algorithm, and seed."""
    grid_shape, k, algo_name, run_id, seed, pop_size, max_iter = args

    prob, env, coords, r_cover, d_min, M, N = build_experiment_problem(
        grid_shape[0], grid_shape[1], k
    )

    opt_cls = ALGORITHMS[algo_name]
    config = OptimizerConfig(
        population_size=pop_size,
        max_iterations=max_iter,
        seed=seed,
        verbose=False,
    )

    t0 = time.perf_counter()
    res = opt_cls(prob, config).optimize()
    dt = time.perf_counter() - t0

    # Kiểm toán tính khả thi: Nếu có bất kỳ trạm nào vi phạm d_min, kích hoạt cơ chế sửa nghiệm an toàn
    if not prob.constraints.check(res.best_solution).feasible:
        res.best_solution = prob.repair(res.best_solution, np.random.default_rng(seed))
        if not prob.constraints.check(res.best_solution).feasible:
            res.best_solution = StationSolution(prob._feasible_witness.candidate_indices.copy())
        res.best_fitness = float(prob.evaluate(res.best_solution, count_fe=False))
        res.best_details = prob.evaluate_with_details(res.best_solution, count_fe=False)

    # Extract coordinates
    best_coords = res.best_solution.coordinates(prob.candidates).tolist()

    return {
        "instance": f"Grid{grid_shape[0]}x{grid_shape[1]}_K{k}",
        "grid": f"{grid_shape[0]}x{grid_shape[1]}",
        "k": k,
        "algorithm": algo_name,
        "run_id": run_id,
        "seed": seed,
        "actual_fe": int(res.evaluations),
        "best_fitness": float(res.best_fitness),
        "coverage_ratio": float(res.best_details["coverage_ratio"]),
        "coverage_pct": float(res.best_details["coverage_percent"]),
        "penalty": float(res.best_details["penalized_fitness"] - res.best_details["fitness"]),
        "is_feasible": bool(res.best_details["feasible"]),
        "runtime_seconds": float(dt),
        "convergence": res.convergence.tolist(),
        "best_coordinates": json.dumps(best_coords),
        "repair_fallback_count": int(prob.repair_fallback_count),
        "random_solution_fallback_count": int(prob.random_solution_fallback_count),
    }


def main():
    print("=" * 75)
    print("  RUNNING HARO & BASELINES GIS BENCHMARK")
    print("  N = 30 independent runs per algorithm (seeds 1..30)")
    print("  Algorithms: HARO (Proposed), ARO, PSO, MVO, GA")
    print("=" * 75)

    instances = [
        ((50, 50), 5),
        ((50, 50), 10),
        ((50, 50), 15),
    ]

    pop_size = 30
    max_iter = 100  # 3,000 FEs per run
    num_runs = 30

    tasks = []
    for grid_shape, k in instances:
        for algo_name in ALGORITHMS.keys():
            for run_idx in range(1, num_runs + 1):
                seed = run_idx  # Seed 1 to 30
                tasks.append((grid_shape, k, algo_name, run_idx, seed, pop_size, max_iter))

    total_tasks = len(tasks)
    print(f"Total optimization tasks to execute: {total_tasks} runs.")
    print(f"Using ProcessPoolExecutor with {os.cpu_count()} CPU cores...\n")

    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    # Run in parallel across CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(run_single_trial, task): task for task in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 30 == 0 or completed == total_tasks:
                print(f"  Progress: {completed}/{total_tasks} runs completed ({completed/total_tasks*100:.1f}%)...")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nAll {total_tasks} runs finished in {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} mins)!")

    # 1. Save raw per-run results
    raw_csv_path = os.path.join(OUTPUT_DIR, "raw_runs_results.csv")
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance", "grid", "k", "algorithm", "run_id", "seed", "actual_fe",
            "best_fitness", "coverage_ratio", "coverage_pct", "penalty",
            "is_feasible", "runtime_seconds", "best_coordinates",
            "repair_fallback_count", "random_solution_fallback_count"
        ])
        for r in sorted(results, key=lambda x: (x["instance"], x["algorithm"], x["run_id"])):
            writer.writerow([
                r["instance"], r["grid"], r["k"], r["algorithm"], r["run_id"], r["seed"],
                r.get("actual_fe", 3030),
                f"{r['best_fitness']:.8f}", f"{r['coverage_ratio']:.8f}", f"{r['coverage_pct']:.4f}",
                f"{r['penalty']:.6f}", r["is_feasible"], f"{r['runtime_seconds']:.4f}",
                r["best_coordinates"], r["repair_fallback_count"],
                r["random_solution_fallback_count"]
            ])
    print(f"Saved raw run results to: {raw_csv_path}")

    # 1b. Save FE Audit Table
    fe_audit_path = os.path.join(OUTPUT_DIR, "bang_fe_audit.csv")
    fe_audit_rows = []
    for a in ALGORITHMS.keys():
        a_runs = [r for r in results if r["algorithm"] == a]
        fe_vals = [r.get("actual_fe", 3030) for r in a_runs]
        fe_audit_rows.append({
            "algorithm": a,
            "total_runs": len(fe_vals),
            "min_fe": int(min(fe_vals)),
            "max_fe": int(max(fe_vals)),
            "mean_fe": float(np.mean(fe_vals)),
            "std_fe": float(np.std(fe_vals)),
            "budget_limit": 3030,
            "budget_overruns": int(sum(f > 3030 for f in fe_vals)),
            "early_stops": int(sum(f < 3030 for f in fe_vals)),
        })
    with open(fe_audit_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fe_audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fe_audit_rows)
    print(f"Saved FE audit table to: {fe_audit_path}")

    # 2. Save convergence traces
    trace_csv_path = os.path.join(OUTPUT_DIR, "convergence_traces.csv")
    with open(trace_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "algorithm", "run_id", "seed", "iteration", "fitness"])
        for r in sorted(results, key=lambda x: (x["instance"], x["algorithm"], x["run_id"])):
            for iter_idx, fit_val in enumerate(r["convergence"]):
                writer.writerow([r["instance"], r["algorithm"], r["run_id"], r["seed"], iter_idx, f"{fit_val:.8f}"])
    print(f"Saved convergence traces to: {trace_csv_path}")

    # 3. Statistical Analysis per Instance (Descriptive + Kruskal-Wallis + Mann-Whitney + Wilcoxon + Holm + Effect Size)
    stats_summary = []
    p_test_summary = []

    instance_names = sorted(list(set(r["instance"] for r in results)))
    algo_names = list(ALGORITHMS.keys())

    for inst in instance_names:
        inst_results = [r for r in results if r["instance"] == inst]

        # Algorithm-wise data
        algo_data = {}
        for algo in algo_names:
            runs = [r for r in inst_results if r["algorithm"] == algo]
            runs = sorted(runs, key=lambda x: x["run_id"])
            fits = np.array([r["best_fitness"] for r in runs])
            covs = np.array([r["coverage_pct"] for r in runs])
            rts = np.array([r["runtime_seconds"] for r in runs])

            q75, q25 = np.percentile(fits, [75, 25])
            iqr = q75 - q25

            cov_q75, cov_q25 = np.percentile(covs, [75, 25])
            cov_iqr = cov_q75 - cov_q25

            algo_data[algo] = {
                "fitness": fits,
                "coverage": covs,
                "runtime": rts,
                "mean_fit": np.mean(fits),
                "std_fit": np.std(fits),
                "median_fit": np.median(fits),
                "iqr_fit": iqr,
                "mean_cov": np.mean(covs),
                "std_cov": np.std(covs),
                "median_cov": np.median(covs),
                "iqr_cov": cov_iqr,
                "best_cov": np.max(covs),
                "mean_rt": np.mean(rts),
            }

            stats_summary.append({
                "instance": inst,
                "algorithm": algo,
                "mean_fitness": float(np.mean(fits)),
                "std_fitness": float(np.std(fits)),
                "median_fitness": float(np.median(fits)),
                "iqr_fitness": float(iqr),
                "mean_cov_pct": float(np.mean(covs)),
                "std_cov_pct": float(np.std(covs)),
                "median_cov_pct": float(np.median(covs)),
                "iqr_cov_pct": float(cov_iqr),
                "best_cov_pct": float(np.max(covs)),
                "mean_runtime_s": float(np.mean(rts)),
            })

        # Kruskal-Wallis Omnibus test across all 5 algorithms
        all_fits = [algo_data[a]["fitness"] for a in algo_names]
        kw_stat, kw_p = stats.kruskal(*all_fits)

        # Pairwise tests: HARO vs each baseline
        haro_fits = algo_data["HARO"]["fitness"]
        baselines = ["ARO", "PSO", "MVO", "GA"]
        mwu_raw = []
        wilc_raw = []
        mwu_stats = []
        wilc_stats = []
        a12_vals = []
        cliffs_vals = []

        for b in baselines:
            b_fits = algo_data[b]["fitness"]
            u1, p_val = stats.mannwhitneyu(haro_fits, b_fits, alternative="two-sided")
            u2 = len(haro_fits) * len(b_fits) - u1
            a12, c_delta = calculate_a12_and_cliffs_delta(haro_fits, b_fits)
            
            diff = haro_fits - b_fits
            w_res = stats.wilcoxon(diff, alternative="two-sided")
            
            mwu_raw.append(p_val)
            mwu_stats.append((u1, u2))
            wilc_raw.append(w_res.pvalue)
            wilc_stats.append(w_res.statistic)
            a12_vals.append(a12)
            cliffs_vals.append(c_delta)

        # Holm-Bonferroni adjustment
        mwu_adj, mwu_sig = holm_bonferroni(mwu_raw, alpha=0.05)
        wilc_adj, wilc_sig = holm_bonferroni(wilc_raw, alpha=0.05)

        for b_idx, b in enumerate(baselines):
            if wilc_adj[b_idx] < 0.05:
                outcome = "Win" if algo_data["HARO"]["mean_fit"] < algo_data[b]["mean_fit"] else "Loss"
            else:
                outcome = "Tie"

            p_test_summary.append({
                "instance": inst,
                "comparison": f"HARO vs {b}",
                "kw_stat": float(kw_stat),
                "kw_p": float(kw_p),
                "u1_stat": float(mwu_stats[b_idx][0]),
                "u2_stat": float(mwu_stats[b_idx][1]),
                "mwu_p_raw": float(mwu_raw[b_idx]),
                "mwu_p_holm": float(mwu_adj[b_idx]),
                "a12": float(a12_vals[b_idx]),
                "cliffs_delta": float(cliffs_vals[b_idx]),
                "wilcoxon_w": float(wilc_stats[b_idx]),
                "wilcoxon_p_raw": float(wilc_raw[b_idx]),
                "wilcoxon_p_holm": float(wilc_adj[b_idx]),
                "outcome": outcome,
            })

    # Save summary stats
    summary_csv_path = os.path.join(OUTPUT_DIR, "bang_thong_ke_mo_ta.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_summary[0].keys()))
        writer.writeheader()
        writer.writerows(stats_summary)
    print(f"Saved descriptive statistics to: {summary_csv_path}")

    # Save hypothesis tests (Per instance)
    tests_csv_path = os.path.join(OUTPUT_DIR, "bang_kiem_dinh_thong_ke.csv")
    with open(tests_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(p_test_summary[0].keys()))
        writer.writeheader()
        writer.writerows(p_test_summary)
    print(f"Saved hypothesis testing results to: {tests_csv_path}")

    # Across-instances Friedman and Wilcoxon
    import pandas as pd
    df_raw = pd.read_csv(raw_csv_path)
    runs_wide = df_raw.pivot(index=["instance", "seed"], columns="algorithm", values="best_fitness")
    f_stat, f_p = stats.friedmanchisquare(runs_wide["HARO"], runs_wide["ARO"], runs_wide["PSO"], runs_wide["MVO"], runs_wide["GA"])
    
    across_rows = []
    across_p_vals = []
    across_w_stats = []
    for b in ["ARO", "PSO", "MVO", "GA"]:
        diff = runs_wide["HARO"] - runs_wide[b]
        w = stats.wilcoxon(diff, alternative="two-sided")
        across_w_stats.append(w.statistic)
        across_p_vals.append(w.pvalue)
        
    across_adj, _ = holm_bonferroni(across_p_vals, alpha=0.05)
    for idx, b in enumerate(["ARO", "PSO", "MVO", "GA"]):
        diff = runs_wide["HARO"] - runs_wide[b]
        wins = np.sum(runs_wide["HARO"].values[:, None] < runs_wide[b].values[None, :])
        ties = np.sum(runs_wide["HARO"].values[:, None] == runs_wide[b].values[None, :])
        a12 = (wins + 0.5 * ties) / (90 * 90)
        delta = 2.0 * a12 - 1.0
        outcome = ("Win" if runs_wide["HARO"].mean() < runs_wide[b].mean() else "Loss") if across_adj[idx] < 0.05 else "Tie"
        across_rows.append({
            "comparison": f"HARO vs {b}",
            "friedman_q": float(f_stat),
            "friedman_p": float(f_p),
            "wilcoxon_w": float(across_w_stats[idx]),
            "wilcoxon_p_raw": float(across_p_vals[idx]),
            "wilcoxon_p_holm": float(across_adj[idx]),
            "a12": float(a12),
            "cliffs_delta": float(delta),
            "outcome": outcome,
        })
    across_csv_path = os.path.join(OUTPUT_DIR, "bang_kiem_dinh_across_instances.csv")
    with open(across_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(across_rows[0].keys()))
        writer.writeheader()
        writer.writerows(across_rows)
    print(f"Saved across-instance hypothesis testing to: {across_csv_path}")

    # 4. Generate Convergence Comparison Plots (by Iteration and by FE)
    print("\nGenerating convergence comparison plots...")
    for inst in instance_names:
        inst_traces = [r for r in results if r["instance"] == inst]
        
        colors = {
            "HARO": "#d62728",
            "ARO": "#1f77b4",
            "PSO": "#2ca02c",
            "MVO": "#ff7f0e",
            "GA": "#9467bd",
        }

        # Plot by iteration
        plt.figure(figsize=(9, 5.5), dpi=300)
        for algo in algo_names:
            runs = [r for r in inst_traces if r["algorithm"] == algo]
            conv_matrix = np.array([r["convergence"] for r in runs])
            mean_conv = np.mean(conv_matrix, axis=0)
            std_conv = np.std(conv_matrix, axis=0)
            iters = np.arange(len(mean_conv))
            line_style = "-" if algo == "HARO" else "--"
            line_width = 2.5 if algo == "HARO" else 1.5
            plt.plot(iters, mean_conv, label=f"{algo} (Mean)", color=colors[algo], linestyle=line_style, linewidth=line_width)
            plt.fill_between(iters, mean_conv - std_conv, mean_conv + std_conv, color=colors[algo], alpha=0.15)
        plt.title(f"Convergence Comparison Across 30 Independent Runs ({inst})", fontsize=12, fontweight="bold")
        plt.xlabel("Iteration", fontsize=11)
        plt.ylabel("Objective Fitness f(X) (Lower is Better)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"convergence_{inst}.png")
        plt.savefig(plot_path)
        plt.close()

        # Plot by FE
        plt.figure(figsize=(9, 5.5), dpi=300)
        for algo in algo_names:
            runs = [r for r in inst_traces if r["algorithm"] == algo]
            conv_matrix = np.array([r["convergence"] for r in runs])
            mean_conv = np.mean(conv_matrix, axis=0)
            std_conv = np.std(conv_matrix, axis=0)
            fes = np.array([30 + t * 30 for t in range(len(mean_conv))])
            line_style = "-" if algo == "HARO" else "--"
            line_width = 2.5 if algo == "HARO" else 1.5
            plt.plot(fes, mean_conv, label=f"{algo} (Mean)", color=colors[algo], linestyle=line_style, linewidth=line_width)
            plt.fill_between(fes, mean_conv - std_conv, mean_conv + std_conv, color=colors[algo], alpha=0.15)
        plt.title(f"Convergence Comparison by Function Evaluations ({inst})", fontsize=12, fontweight="bold")
        plt.xlabel("Function Evaluations (FE)", fontsize=11)
        plt.ylabel("Objective Fitness f(X) (Lower is Better)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        fe_plot_path = os.path.join(OUTPUT_DIR, f"fe_convergence_{inst}.png")
        plt.savefig(fe_plot_path)
        plt.close()
        print(f"  Saved iteration & FE convergence plots for {inst}")

        # Plot Heatmap for best HARO run
        haro_runs = [r for r in inst_traces if r["algorithm"] == "HARO"]
        best_haro = min(haro_runs, key=lambda x: x["best_fitness"])
        best_coords = np.array(json.loads(best_haro["best_coordinates"]))
        grid_shape = (50, 50)
        k_val = best_haro["k"]
        _, env, _, r_cov, _, _, _ = build_experiment_problem(grid_shape[0], grid_shape[1], k_val)
        heatmap_path = os.path.join(OUTPUT_DIR, f"heatmap_{inst}.png")
        from visualization.plots import plot_heatmap_with_stations
        plot_heatmap_with_stations(
            pollution_grid=env.concentration,
            stations=best_coords,
            R_cover=r_cov,
            sources=env.sources,
            coverage_pct=best_haro["coverage_pct"],
            title=f"Mạng lưới trạm tối ưu HARO ({inst}) - Độ phủ: {best_haro['coverage_pct']:.2f}%",
            save_path=heatmap_path,
        )
        print(f"  Saved heatmap plot for {inst}: {heatmap_path}")

    # Generate combined 3-in-1 FE convergence plot (K=5, K=10, K=15)
    print("Generating combined 3-in-1 FE convergence plot...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), dpi=300)
    for idx, inst in enumerate(["Grid50x50_K5", "Grid50x50_K10", "Grid50x50_K15"]):
        ax = axes[idx]
        inst_traces = [r for r in results if r["instance"] == inst]
        for algo in algo_names:
            runs = [r for r in inst_traces if r["algorithm"] == algo]
            conv_matrix = np.array([r["convergence"] for r in runs])
            mean_conv = np.mean(conv_matrix, axis=0)
            std_conv = np.std(conv_matrix, axis=0)
            fes = np.array([30 + t * 30 for t in range(len(mean_conv))])
            line_style = "-" if algo == "HARO" else "--"
            line_width = 2.5 if algo == "HARO" else 1.5
            ax.plot(fes, mean_conv, label=f"{algo}", color=colors[algo], linestyle=line_style, linewidth=line_width)
            ax.fill_between(fes, mean_conv - std_conv, mean_conv + std_conv, color=colors[algo], alpha=0.15)
        k_val = inst.split("_K")[1]
        ax.set_title(f"Lưới 50x50 | K={k_val} trạm", fontsize=12, fontweight="bold")
        ax.set_xlabel("Đánh giá hàm mục tiêu (FE)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Hàm thích nghi f(X) (Càng âm càng tốt)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right")
    plt.suptitle("So sánh hội tụ của 5 thuật toán qua 30 lượt chạy độc lập (Lưới 50x50, K=5, 10, 15)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fe_comb_path = os.path.join(OUTPUT_DIR, "fe_convergence_Grid50x50_K5-K10-K15.png")
    plt.savefig(fe_comb_path)
    plt.close()
    print(f"  Saved combined FE convergence plot to: {fe_comb_path}")

    # Generate combined 3-in-1 Heatmap plot (K=5, K=10, K=15)
    print("Generating combined 3-in-1 Heatmap plot...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=300)
    for idx, inst in enumerate(["Grid50x50_K5", "Grid50x50_K10", "Grid50x50_K15"]):
        ax = axes[idx]
        inst_traces = [r for r in results if r["instance"] == inst]
        haro_runs = [r for r in inst_traces if r["algorithm"] == "HARO"]
        best_haro = min(haro_runs, key=lambda x: x["best_fitness"])
        best_coords = np.array(json.loads(best_haro["best_coordinates"]))
        k_val = best_haro["k"]
        _, env, mask, r_cov, _, _, _ = build_experiment_problem(50, 50, k_val)
        
        from visualization.plots import AQI_CMAP
        im = ax.imshow(env.concentration.T, origin="lower", cmap=AQI_CMAP, aspect="equal")
        if mask is not None:
            forbidden_coords = np.argwhere(mask == 0)
            if len(forbidden_coords) > 0:
                ax.scatter(forbidden_coords[:, 0], forbidden_coords[:, 1], color="#333333", s=10, alpha=0.3, label="Vùng cấm")
        for sx, sy in best_coords:
            c = plt.Circle((sx, sy), r_cov, color="#39ff14", fill=True, alpha=0.25, linestyle="--", linewidth=1.5)
            ax.add_patch(c)
            ax.plot(sx, sy, "^", color="#39ff14", markersize=9, markeredgecolor="black", markeredgewidth=1.2)
        
        ax.set_title(f"HARO K={k_val} (Độ phủ: {best_haro['coverage_pct']:.2f}%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("X (ô lưới)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Y (ô lưới)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)
    plt.suptitle("Bản đồ phân bố trạm tối ưu của HARO trên lưới 50x50 (K = 5, 10, 15)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    heat_comb_path = os.path.join(OUTPUT_DIR, "heatmap_Grid50x50_K5-K10-K15.png")
    plt.savefig(heat_comb_path)
    plt.close()
    print(f"  Saved combined heatmap plot to: {heat_comb_path}")

    # Save summary table bang_tong_hop_ket_qua.csv
    summary_path = os.path.join(OUTPUT_DIR, "bang_tong_hop_ket_qua.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "grid", "K", "algorithm", "pop_size", "T", "R_cover", "d_min", "n_runs",
            "fitness_mean", "fitness_std", "fitness_best",
            "coverage_pct_mean", "coverage_pct_std", "coverage_pct_best",
            "time_s_per_run_mean", "time_s_total",
            "repair_fallback_count_total", "random_solution_fallback_count_total"
        ])
        for inst in instance_names:
            inst_results = [r for r in results if r["instance"] == inst]
            k_val = inst_results[0]["k"]
            grid_val = inst_results[0]["grid"]
            for algo in algo_names:
                runs = [r for r in inst_results if r["algorithm"] == algo]
                fits = np.array([r["best_fitness"] for r in runs])
                covs = np.array([r["coverage_pct"] for r in runs])
                rts = np.array([r["runtime_seconds"] for r in runs])
                rf_count = sum(r.get("repair_fallback_count", 0) for r in runs)
                rand_count = sum(r.get("random_solution_fallback_count", 0) for r in runs)
                writer.writerow([
                    grid_val, k_val, algo, pop_size, max_iter, 8.0, 10.0, len(runs),
                    round(float(np.mean(fits)), 4), round(float(np.std(fits)), 4), round(float(np.min(fits)), 4),
                    round(float(np.mean(covs)), 2), round(float(np.std(covs)), 2), round(float(np.max(covs)), 2),
                    round(float(np.mean(rts)), 2), round(float(np.sum(rts)), 2),
                    rf_count, rand_count
                ])
    print(f"Saved summary results table to: {summary_path}")


    # Print Summary Tables in Terminal
    print("\n" + "=" * 95)
    print(f"{'Instance':<15} | {'Algorithm':<8} | {'Mean Fitness ± Std':<22} | {'Median (IQR)':<18} | {'Mean Cov (%)':<12} | {'Time (s)':<8}")
    print("-" * 95)
    for s in stats_summary:
        fit_str = f"{s['mean_fitness']:.4f} ± {s['std_fitness']:.4f}"
        med_str = f"{s['median_fitness']:.4f} ({s['iqr_fitness']:.4f})"
        print(f"{s['instance']:<15} | {s['algorithm']:<8} | {fit_str:<22} | {med_str:<18} | {s['mean_cov_pct']:.2f}%{'':<6} | {s['mean_runtime_s']:.2f}")

    print("\n" + "=" * 105)
    print(f"{'Instance':<15} | {'Comparison':<16} | {'KW p-val':<10} | {'Wilc p_raw':<12} | {'Holm p_adj':<12} | {'A12':<7} | {'Cliff d':<8} | {'W/T/L'}")
    print("-" * 105)
    for t in p_test_summary:
        print(f"{t['instance']:<15} | {t['comparison']:<16} | {t['kw_p']:.4e} | {t['wilcoxon_p_raw']:.4e}   | {t['wilcoxon_p_holm']:.4e}   | {t['a12']:.4f} | {t['cliffs_delta']:+.4f} | {t['outcome']}")
    print("=" * 105)


if __name__ == "__main__":
    main()

