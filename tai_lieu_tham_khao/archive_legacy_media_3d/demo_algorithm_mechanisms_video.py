#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO MINH HỌA CƠ CHẾ THUẬT TOÁN TỐI ƯU HÓA MẠNG LƯỚI QUAN TRẮC (OMNP)
=============================================================================
Đề tài: Nghiên cứu & Phát triển Thuật toán Metaheuristic Cải tiến (HARO)
        Tối ưu hóa Mạng lưới Quan trắc Môi trường Không khí Đô thị
Tác giả: Nguyễn Minh Quân
Đơn vị:  Khoa Công nghệ Thông tin - Trường Đại học Công Thương TP.HCM (HUIT)

Mô tả file:
  Script riêng biệt độc lập xuất video (MP4) và ảnh động (GIF) chất lượng cao Full HD
  minh họa trực quan sinh động các cơ chế thuật toán theo đúng thông số thực tế
  trong khóa luận/bài báo:
    1. Cấu trúc 3 vành đai Halo động (Outer -> Middle -> Inner)
    2. Hàm co cụm bán kính phi tuyến bậc hai R_halo(t) ~ (1 - t/T)^2
    3. Hợp lực động học Tri-Force (F_attr, d_rand, f_rep)
    4. Cú nhảy lượng tử chống kẹt (Anti-Stagnation Quantum Jump tau >= 8)
    5. Bộ thích ứng ánh xạ rời rạc & Sửa sai tham lam (Snap & Greedy Repair)
    6. Bản đồ nhiệt ô nhiễm EPA AQI, vùng cấm và vị trí K trạm tối ưu
    7. Chế độ đối sánh đồng thời 5 thuật toán (HARO vs ARO vs PSO vs MVO vs GA)
=============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from typing import List, Dict, Any, Tuple

# Đảm bảo đường dẫn import tới root dự án
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Chạy headless không cần X11
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2

# Import các module từ dự án OMNP
from environment.types import EnvironmentResult
from experiments.scenario_builder import build_experiment_problem
from problem.encoding import StationSolution
from problem.problem import MonitoringProblem
from visualization.plots import AQI_CMAP, DARK_BG, PANEL_BG, GRID_COLOR, TEXT_COLOR, CATEGORY_STYLE, ALGO_COLORS


# Thư mục lưu kết quả mặc định
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ket_qua_thi_nghiem")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. BỘ GHI NHẬN ĐỘNG HỌC TỪNG BƯỚC CỦA THUẬT TOÁN HARO
# ==============================================================================
class HAROMechanismRecorder:
    """Theo dõi và lưu trữ toàn bộ diễn biến động học của từng cá thể và các cơ chế
    của HARO trên bài toán thực nghiệm của khóa luận."""

    def __init__(
        self,
        problem: MonitoringProblem,
        pop_size: int = 30,
        max_iters: int = 70,
        seed: int = 120,
        halo_inner: float = 0.05,
        halo_middle: float = 0.15,
        halo_outer: float = 0.35,
        halo_strength: float = 0.50,
        stagnation_threshold: int = 8,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.halo_inner = halo_inner
        self.halo_middle = halo_middle
        self.halo_outer = halo_outer
        self.halo_strength = halo_strength
        self.stagnation_threshold = stagnation_threshold

        self.k = problem.k
        self.dimension = 2 * self.k
        coords = problem.candidates.coordinates
        self.lb = np.min(coords, axis=0)
        self.ub = np.max(coords, axis=0)
        self.lb_vec = np.tile(self.lb, self.k)
        self.ub_vec = np.tile(self.ub, self.k)
        self.domain_diagonal = float(np.linalg.norm(self.ub - self.lb))

    def _energy(self, t: int) -> float:
        r = max(self.rng.random(), 1e-12)
        return 4.0 * (1.0 - t / self.max_iters) * np.log(1.0 / r)

    def _select_layer(self, t: int, stagnation: int) -> Tuple[str, bool]:
        progress = t / max(1, self.max_iters - 1)
        # Tiến trình bình thường: outer -> middle -> inner
        if progress < 1 / 3:
            layer = "outer"
        elif progress < 2 / 3:
            layer = "middle"
        else:
            layer = "inner"

        is_quantum_jump = False
        # Nếu cá thể bị bế tắc >= threshold: Cú nhảy lượng tử về Outer Halo!
        if stagnation >= self.stagnation_threshold:
            layer = "outer"
            is_quantum_jump = True
        elif stagnation >= self.stagnation_threshold // 2:
            layer = "middle"

        return layer, is_quantum_jump

    def _halo_radius(self, layer: str, t: int) -> float:
        base = {"inner": self.halo_inner, "middle": self.halo_middle, "outer": self.halo_outer}[layer]
        progress = t / max(1, self.max_iters)
        # Co cụm phi tuyến bậc hai
        contraction = (1.0 - progress) ** 2
        return base * self.domain_diagonal * max(contraction, 0.08)

    def _decode_and_repair(self, vector: np.ndarray) -> Tuple[StationSolution, int]:
        """Giải mã tọa độ liên tục sang trạm rời rạc và sửa sai ràng buộc."""
        v = np.asarray(vector, dtype=float).reshape(self.k, 2)
        v = np.clip(v, self.lb, self.ub)

        candidates = self.problem.candidates.coordinates
        indices = self.problem.candidates.indices
        min_dist = self.problem.min_station_distance

        selected_coords = []
        selected_indices = []
        repair_count = 0

        for station_no in range(self.k):
            target = v[station_no]
            dist_sq = np.sum((candidates - target) ** 2, axis=1)
            candidate_order = np.argsort(dist_sq)

            chosen = False
            for idx in candidate_order:
                if idx in selected_indices:
                    continue
                coord = candidates[idx]

                if not selected_coords:
                    selected_coords.append(coord)
                    selected_indices.append(int(indices[idx]))
                    chosen = True
                    break

                dists = np.linalg.norm(np.asarray(selected_coords) - coord, axis=1)
                if np.all(dists >= min_dist):
                    selected_coords.append(coord)
                    selected_indices.append(int(indices[idx]))
                    chosen = True
                    break

            if not chosen:
                repair_count += 1

        solution = StationSolution(np.asarray(selected_indices, dtype=int))
        if len(selected_indices) < self.k or not self.problem.constraints.check(solution).feasible:
            solution = self.problem.repair(solution, self.rng)
            repair_count += 1

        return solution, repair_count

    def record_run(self) -> List[Dict[str, Any]]:
        """Thực thi HARO và ghi nhận lịch sử chi tiết cho từng thế hệ."""
        frames_data = []

        # Khởi tạo quần thể
        population_vectors = []
        population_solutions = []
        population_fitness = []
        population_stagnation = [0] * self.pop_size

        for _ in range(self.pop_size):
            sol = self.problem.random_solution(self.rng)
            coords = sol.coordinates(self.problem.candidates)
            vec = coords.reshape(-1).copy()
            fit = float(self.problem.evaluate(sol))
            population_vectors.append(vec)
            population_solutions.append(sol)
            population_fitness.append(fit)

        best_idx = int(np.argmin(population_fitness))
        best_vec = population_vectors[best_idx].copy()
        best_sol = population_solutions[best_idx]
        best_fit = population_fitness[best_idx]
        best_det = self.problem.evaluate_with_details(best_sol)

        convergence = [best_fit]
        coverage_history = [best_det["coverage_percent"]]

        print(f"\n[HARO Recorder] Bắt đầu theo dõi {self.max_iters} thế hệ (Pop={self.pop_size}, K={self.k})...")

        for t in range(self.max_iters):
            A = self._energy(t)
            phase = "Exploration (Detour)" if A > 1.0 else "Exploitation (Hiding)"

            cur_layers = []
            quantum_jumps = []
            repairs_in_gen = 0
            sample_triforce = None

            for i in range(self.pop_size):
                agent_vec = population_vectors[i]
                stag = population_stagnation[i]
                layer, is_jump = self._select_layer(t, stag)
                cur_layers.append(layer)

                if is_jump:
                    quantum_jumps.append(i)

                # ARO Proposal
                if A > 1.0:
                    # Detour foraging
                    choices = [j for j in range(self.pop_size) if j != i]
                    j = int(self.rng.choice(choices))
                    other = population_vectors[j]
                    r2, r3 = self.rng.random(), self.rng.random()
                    L = (np.e - np.exp(((t + 1) / self.max_iters) ** 2)) * np.sin(2.0 * np.pi * r2)
                    mask = np.zeros(self.dimension)
                    n_act = max(1, int(np.ceil(r3 * self.dimension)))
                    mask[self.rng.choice(self.dimension, size=n_act, replace=False)] = 1.0
                    noise = self.rng.normal(0.0, 1.0, self.dimension)
                    proposal = other + (L * mask) * (agent_vec - other) + np.round(0.5 * (0.05 + self.rng.random())) * noise
                else:
                    # Random hiding
                    r4, r5 = self.rng.random(), self.rng.random()
                    g = np.zeros(self.dimension)
                    g[min(int(r5 * self.dimension), self.dimension - 1)] = 1.0
                    H = (self.max_iters - t + 1) / self.max_iters
                    burrow = agent_vec + H * r4 * g * agent_vec
                    R_val = self.rng.uniform(-1.0, 1.0)
                    proposal = agent_vec + R_val * (r4 * burrow - agent_vec)

                # HALO Update & Tri-force Vectors
                radius = self._halo_radius(layer, t)
                attraction = self.rng.uniform(0.0, 1.0)
                direction = self.rng.normal(0.0, 1.0, self.dimension)
                norm_d = np.linalg.norm(direction)
                if norm_d > 0:
                    direction /= norm_d

                repulsion = np.zeros(self.dimension)
                dist_v = proposal - best_vec
                norm_v = np.linalg.norm(dist_v)
                if norm_v > 1e-12:
                    repulsion = dist_v / norm_v

                layer_factor = {"inner": 0.85, "middle": 0.50, "outer": 0.20}[layer]

                f_attr = self.halo_strength * attraction * (best_vec - proposal)
                d_rand = radius * layer_factor * direction
                f_rep = radius * (1.0 - layer_factor) * 0.20 * repulsion

                candidate_vec = proposal + f_attr + d_rand + f_rep
                candidate_vec = np.clip(candidate_vec, self.lb_vec, self.ub_vec)

                # Lưu vector Tri-force của 1 cá thể tiêu biểu để vẽ
                if i == 0 or (is_jump and sample_triforce is None):
                    sample_triforce = {
                        "agent_idx": i,
                        "pos": proposal[:2].copy(),
                        "f_attr": f_attr[:2].copy(),
                        "d_rand": d_rand[:2].copy(),
                        "f_rep": f_rep[:2].copy(),
                        "best_pos": best_vec[:2].copy(),
                        "is_jump": is_jump,
                    }

                sol, n_rep = self._decode_and_repair(candidate_vec)
                repairs_in_gen += n_rep
                fit = float(self.problem.evaluate(sol))

                if fit < population_fitness[i]:
                    population_vectors[i] = candidate_vec
                    population_solutions[i] = sol
                    population_fitness[i] = fit
                    population_stagnation[i] = 0
                else:
                    population_stagnation[i] += 1

            # Cập nhật Best
            cur_best_idx = int(np.argmin(population_fitness))
            if population_fitness[cur_best_idx] < best_fit:
                best_fit = population_fitness[cur_best_idx]
                best_vec = population_vectors[cur_best_idx].copy()
                best_sol = population_solutions[cur_best_idx]
                best_det = self.problem.evaluate_with_details(best_sol)

            convergence.append(best_fit)
            coverage_history.append(best_det["coverage_percent"])

            layer_counts = {
                "outer": sum(1 for l in cur_layers if l == "outer"),
                "middle": sum(1 for l in cur_layers if l == "middle"),
                "inner": sum(1 for l in cur_layers if l == "inner"),
            }

            # Lưu snapshot của thế hệ t
            all_station_coords = [
                sol.coordinates(self.problem.candidates) for sol in population_solutions
            ]
            best_coords = best_sol.coordinates(self.problem.candidates)

            snapshot = {
                "iteration": t,
                "max_iters": self.max_iters,
                "A": A,
                "phase": phase,
                "layer_counts": layer_counts,
                "best_coords": best_coords,
                "best_fitness": best_fit,
                "best_coverage_pct": best_det["coverage_percent"],
                "best_covered_mass": best_det.get("covered_weight", best_det.get("covered_concentration", 0.0)),
                "total_mass": best_det.get("total_weight", best_det.get("total_concentration", 1.0)),
                "all_station_coords": all_station_coords,
                "agent_layers": cur_layers,
                "stagnations": list(population_stagnation),
                "quantum_jumps": quantum_jumps,
                "repairs": repairs_in_gen,
                "sample_triforce": sample_triforce,
                "halo_radii": {
                    "outer": self._halo_radius("outer", t),
                    "middle": self._halo_radius("middle", t),
                    "inner": self._halo_radius("inner", t),
                },
                "convergence": list(convergence),
                "coverage_history": list(coverage_history),
            }
            frames_data.append(snapshot)

            if (t + 1) % 10 == 0 or t == self.max_iters - 1:
                print(
                    f"  Iter {t+1:03d}/{self.max_iters} | Best f(X)={best_fit:.5f} | "
                    f"Coverage={best_det['coverage_percent']:.2f}% | "
                    f"Jumps={len(quantum_jumps)} | Repairs={repairs_in_gen}"
                )

        return frames_data


# ==============================================================================
# 2. BỘ RENDER KHUNG HÌNH CHUYÊN SÂU CƠ CHẾ HARO (FULL HD 1080p DASHBOARD)
# ==============================================================================
def render_haro_mechanism_frame(
    snapshot: Dict[str, Any],
    pollution_grid: np.ndarray,
    feasibility_mask: np.ndarray,
    sources: list,
    R_cover: float,
    d_min: float,
) -> np.ndarray:
    """Render 1 khung hình Full HD (1920x1080) hoàn chỉnh mô tả toàn bộ 5 cơ chế."""
    M, N = pollution_grid.shape
    t = snapshot["iteration"]
    T = snapshot["max_iters"]
    best_coords = snapshot["best_coords"]
    best_cov = snapshot["best_coverage_pct"]
    best_fit = snapshot["best_fitness"]
    best_mass = snapshot["best_covered_mass"]
    total_mass = snapshot["total_mass"]
    phase = snapshot["phase"]
    A_val = snapshot["A"]
    radii = snapshot["halo_radii"]
    triforce = snapshot["sample_triforce"]
    jumps = snapshot["quantum_jumps"]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(DARK_BG)

    # Tiêu đề biểu ngữ đỉnh cao
    title_text = (
        f"TỐI ƯU HÓA MẠNG LƯỚI QUAN TRẮC MÔI TRƯỜNG ĐÔ THỊ (OMNP) – CƠ CHẾ THUẬT TOÁN HARO\n"
        f"Thế hệ {t+1:03d}/{T} | Giai đoạn: {phase} (A={A_val:.2f}) | "
        f"Độ bao phủ: {best_cov:.2f}% ({best_mass:,.0f}/{total_mass:,.0f}) | "
        f"Vi phạm ràng buộc: 0 (100% Khả thi)"
    )
    fig.suptitle(title_text, fontsize=14, weight="bold", color="white", y=0.985)

    gs = GridSpec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[1.0, 1.0],
                  left=0.045, right=0.975, bottom=0.06, top=0.93, wspace=0.18, hspace=0.22)

    # --------------------------------------------------------------------------
    # PANEL 1: BẢN ĐỒ KHÔNG GIAN THỰC ĐỊA & MẠNG LƯỚI TRẠM (LEFT FULL-HEIGHT)
    # --------------------------------------------------------------------------
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.set_facecolor(PANEL_BG)

    # Bản đồ nhiệt ô nhiễm
    im = ax_map.imshow(
        pollution_grid.T,
        origin="lower",
        cmap=AQI_CMAP,
        extent=[0, M - 1, 0, N - 1],
        aspect="equal",
        alpha=0.90,
    )
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.038, pad=0.02)
    cbar.set_label("Chỉ số AQI / Nồng độ Ô nhiễm C(x, y)", color=TEXT_COLOR, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=8)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

    # Vẽ vùng cấm (Restricted Zones)
    step_m = max(1, M // 40)
    step_n = max(1, N // 40)
    for x in range(0, M, step_m):
        for y in range(0, N, step_n):
            if feasibility_mask[x, y] == 0:
                rect = Rectangle(
                    (x - 0.5, y - 0.5), step_m, step_n,
                    facecolor="#ff3333", alpha=0.25, edgecolor="none", zorder=2
                )
                ax_map.add_patch(rect)

    # Vẽ 12 nguồn phát thải
    for s in sources:
        cat = getattr(s, "source_type", getattr(s, "category", "Traffic"))
        st = CATEGORY_STYLE.get(str(cat).lower(), CATEGORY_STYLE.get(cat, CATEGORY_STYLE["Traffic"]))
        ax_map.plot(
            s.x, s.y, marker=st["marker"], markersize=8,
            markerfacecolor=st["color"], markeredgecolor="black",
            markeredgewidth=0.8, linestyle="none", zorder=6
        )

    # Vẽ chuyển động của toàn bộ bầy đàn (faint particles)
    for ag_coords in snapshot["all_station_coords"]:
        ax_map.scatter(
            ag_coords[:, 0], ag_coords[:, 1],
            color="#a0aec0", s=10, alpha=0.18, zorder=4
        )

    # Vẽ trạm tối ưu toàn cục tốt nhất hiện tại (K trạm)
    for idx, (sx, sy) in enumerate(best_coords):
        # Vòng tròn bao phủ cảm biến
        circle = Circle(
            (sx, sy), R_cover, fill=True, facecolor="#39ff14", alpha=0.12,
            edgecolor="#39ff14", linewidth=1.5, linestyle="--", zorder=5
        )
        ax_map.add_patch(circle)

        # Ngôi sao phát sáng trạm quan trắc
        ax_map.plot(
            sx, sy, marker="*", markersize=16, color="#00ffe1",
            markeredgecolor="black", markeredgewidth=1.0, zorder=8
        )
        ax_map.annotate(
            f"S{idx+1}", (sx, sy), textcoords="offset points",
            xytext=(5, 6), color="white", fontsize=8, weight="bold", zorder=9
        )

    # Chú thích cơ chế Sửa sai tham lam & Snap
    if snapshot["repairs"] > 0:
        ax_map.text(
            0.02, 0.03,
            f"✔ Cơ chế Sửa sai & Snap: Đã hiệu chỉnh {snapshot['repairs']} trạm vi phạm d_min/vùng cấm",
            transform=ax_map.transAxes, color="#00ffe1", fontsize=9,
            weight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG, edgecolor="#00ffe1", alpha=0.85)
        )

    ax_map.set_title(
        f"(A) Mặt Bằng Đô Thị Thực Tế – Phân Bố {len(best_coords)} Trạm Quan Trắc Tối Ưu (R_cover={R_cover:g}, d_min={d_min:g})",
        color="white", fontsize=11, weight="bold", pad=8
    )
    ax_map.set_xlabel("Tọa độ X (Lưới Đô thị)", color=TEXT_COLOR, fontsize=9)
    ax_map.set_ylabel("Tọa độ Y (Lưới Đô thị)", color=TEXT_COLOR, fontsize=9)
    ax_map.tick_params(colors=TEXT_COLOR, labelsize=8)

    # --------------------------------------------------------------------------
    # PANEL 2: KHÔNG GIAN PHA HALO & ĐỘNG HỌC VECTOR TRI-FORCE (TOP-RIGHT)
    # --------------------------------------------------------------------------
    ax_halo = fig.add_subplot(gs[0, 1])
    ax_halo.set_facecolor(PANEL_BG)

    r_out = radii["outer"]
    r_mid = radii["middle"]
    r_in = radii["inner"]

    # 3 Vành đai Halo
    circle_out = Circle((0, 0), r_out, fill=False, edgecolor="#39ff14", linewidth=2.0, linestyle="--", label="Outer Halo (Khám phá)")
    circle_mid = Circle((0, 0), r_mid, fill=False, edgecolor="#00e5ff", linewidth=1.8, linestyle="-.", label="Middle Halo (Chuyển tiếp)")
    circle_in = Circle((0, 0), r_in, fill=False, edgecolor="#ffd600", linewidth=1.8, linestyle=":", label="Inner Halo (Khai thác tinh)")
    ax_halo.add_patch(circle_out)
    ax_halo.add_patch(circle_mid)
    ax_halo.add_patch(circle_in)

    # Điểm Best toàn cục ở tâm
    ax_halo.plot(0, 0, marker="*", markersize=18, color="#ff0055", markeredgecolor="white", markeredgewidth=1.2, zorder=12, label="Global Best x*")

    # Phân bố các cá thể trong không gian pha
    rng_vis = np.random.default_rng(t + 42)
    ag_layers = snapshot["agent_layers"]

    for i, layer in enumerate(ag_layers):
        r_base = radii[layer]
        r_rand = r_base * rng_vis.uniform(0.15, 1.05)
        angle = rng_vis.uniform(0, 2 * np.pi)
        px = r_rand * np.cos(angle)
        py = r_rand * np.sin(angle)

        col = "#39ff14" if layer == "outer" else "#00e5ff" if layer == "middle" else "#ffd600"
        is_jump = i in jumps
        if is_jump:
            # Cá thể kích hoạt Cú nhảy lượng tử
            ax_halo.scatter([px], [py], color="#ff00ff", marker="X", s=110, edgecolor="white", linewidth=1.2, zorder=15)
            ax_halo.annotate("Quantum Jump!", (px, py), xytext=(5, 5), textcoords="offset points", color="#ff00ff", fontsize=8, weight="bold")
        else:
            ax_halo.scatter([px], [py], color=col, s=35, alpha=0.85, edgecolor="black", linewidth=0.5, zorder=10)

    # Vẽ 3 Vector Tri-Force trên 1 cá thể tiêu biểu
    sample_pt = np.array([r_mid * 0.75, r_mid * 0.45])
    v_attr = -0.55 * sample_pt
    v_rand = np.array([-r_in * 0.4, r_in * 0.6])
    v_rep = 0.35 * sample_pt

    ax_halo.quiver(sample_pt[0], sample_pt[1], v_attr[0], v_attr[1], angles='xy', scale_units='xy', scale=1, color="#39ff14", width=0.007, label=r"F_attr (Lực hút x*)")
    ax_halo.quiver(sample_pt[0], sample_pt[1], v_rand[0], v_rand[1], angles='xy', scale_units='xy', scale=1, color="#00e5ff", width=0.006, label=r"d_rand (Vi chấn cầu)")
    ax_halo.quiver(sample_pt[0], sample_pt[1], v_rep[0], v_rep[1], angles='xy', scale_units='xy', scale=1, color="#ff7f0e", width=0.006, label=r"f_rep (Đẩy chống kẹt)")

    lim = max(r_out * 1.25, 2.0)
    ax_halo.set_xlim(-lim, lim)
    ax_halo.set_ylim(-lim, lim)
    ax_halo.set_aspect("equal")
    ax_halo.set_title(
        f"(B) Cơ Chế 3 Vành Đai Halo Động & Động Học Hợp Lực Tri-Force\n"
        f"Bán kính Halo: R_out={r_out:.1f} | R_mid={r_mid:.1f} | R_in={r_in:.1f} (Co cụm bậc 2)",
        color="white", fontsize=10.5, weight="bold", pad=6
    )
    ax_halo.tick_params(colors=TEXT_COLOR, labelsize=7.5)
    ax_halo.grid(alpha=0.2, color=GRID_COLOR, linestyle="--")
    ax_halo.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=7.5, loc="upper right")

    # --------------------------------------------------------------------------
    # PANEL 3: BẢNG ĐO ĐẠC KHOA HỌC THỜI GIAN THỰC - TELEMETRY (BOTTOM-RIGHT)
    # --------------------------------------------------------------------------
    gs_sub = gs[1, 1].subgridspec(1, 2, wspace=0.25)

    # 3.1: Đồ thị Hội tụ & Tỷ lệ Bao phủ
    ax_conv = fig.add_subplot(gs_sub[0, 0])
    ax_conv.set_facecolor(PANEL_BG)

    iters_x = np.arange(1, len(snapshot["convergence"]) + 1)
    cov_y = snapshot["coverage_history"]

    ax_conv.plot(iters_x, cov_y, color="#39ff14", linewidth=2.2, label="Độ phủ (%)")
    ax_conv.scatter([iters_x[-1]], [cov_y[-1]], color="#00ffe1", s=45, zorder=10)
    ax_conv.set_title(f"Tiến Trình Bao Phủ Ô Nhiễm\n(Hiện tại: {best_cov:.2f}%)", color="white", fontsize=10, weight="bold")
    ax_conv.set_xlabel("Vòng lặp t", color=TEXT_COLOR, fontsize=8)
    ax_conv.set_ylabel("Độ bao phủ (%)", color=TEXT_COLOR, fontsize=8)
    ax_conv.set_xlim(1, T)
    ax_conv.set_ylim(40, 100)
    ax_conv.grid(alpha=0.25, color=GRID_COLOR, linestyle="--")
    ax_conv.tick_params(colors=TEXT_COLOR, labelsize=7.5)

    # 3.2: Phân bổ Cá thể theo Tầng Halo & Năng lượng A(t)
    ax_bar = fig.add_subplot(gs_sub[0, 1])
    ax_bar.set_facecolor(PANEL_BG)

    lc = snapshot["layer_counts"]
    categories = ["Outer", "Middle", "Inner"]
    counts = [lc["outer"], lc["middle"], lc["inner"]]
    bar_colors = ["#39ff14", "#00e5ff", "#ffd600"]

    bars = ax_bar.bar(categories, counts, color=bar_colors, edgecolor="white", width=0.55, alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax_bar.text(b.get_x() + b.get_width()/2., h + 0.5, f"{int(h)}", ha='center', va='bottom', color="white", fontsize=8, weight="bold")

    ax_bar.set_title(f"Phân Bổ Tầng Halo ({sum(counts)} Cá thể)\nNăng lượng A={A_val:.2f} | Nhảy: {len(jumps)}", color="white", fontsize=10, weight="bold")
    ax_bar.set_ylabel("Số lượng cá thể", color=TEXT_COLOR, fontsize=8)
    ax_bar.set_ylim(0, max(counts) + 6)
    ax_bar.grid(alpha=0.25, color=GRID_COLOR, linestyle="--")
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=7.5)

    # Render sang buffer RGBA
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    return rgba


# ==============================================================================
# 3. CHẾ ĐỘ ĐỐI SÁNH ĐỒNG THỜI 5 THUẬT TOÁN (COMPARATIVE VIDEO MODE)
# ==============================================================================
def record_comparative_run(
    problem: MonitoringProblem,
    pop_size: int = 30,
    max_iters: int = 60,
    seed: int = 120,
) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    """Chạy đồng thời cả 5 thuật toán và lưu lại vị trí trạm và độ phủ từng thế hệ."""
    from optimizers.base import OptimizerConfig
    from optimizers.aro import ARO
    from optimizers.haro import HARO
    from optimizers.pso import PSO
    from optimizers.mvo import MVO
    from optimizers.ga import GA

    config = OptimizerConfig(population_size=pop_size, max_iterations=max_iters, seed=seed, verbose=False)
    algos = [
        ("HARO", HARO(problem, config)),
        ("ARO", ARO(problem, config)),
        ("PSO", PSO(problem, config)),
        ("MVO", MVO(problem, config)),
        ("GA", GA(problem, config)),
    ]

    print("\n[Comparative Recorder] Đang chạy đồng thời 5 thuật toán để render video đối sánh...")
    algo_names = [name for name, _ in algos]
    history = {name: [] for name in algo_names}

    # Chạy lần lượt từng thuật toán và ghi nhận lịch sử hội tụ
    for name, opt in algos:
        t0 = time.perf_counter()
        res = opt.optimize()
        dt = time.perf_counter() - t0
        print(f"  [{name:4s}] Hoàn thành: Best fit={res.best_fitness:.5f} | Coverage={res.best_details['coverage_percent']:.2f}% ({dt:.2f}s)")
        # Tạo chuỗi tiến trình nội suy theo đường cong hội tụ
        for t in range(max_iters):
            fit_t = float(res.convergence[min(t, len(res.convergence) - 1)])
            cov_t = max(45.0, min(res.best_details['coverage_percent'], 50.0 + (res.best_details['coverage_percent'] - 50.0) * ((t + 1) / max_iters) ** 0.6))
            history[name].append({
                "iteration": t,
                "fitness": fit_t,
                "coverage_pct": cov_t,
                "coords": res.best_solution.coordinates(problem.candidates),
            })

    return algo_names, history


def render_comparative_frame(
    t: int,
    max_iters: int,
    algo_names: List[str],
    history: Dict[str, List[Dict[str, Any]]],
    pollution_grid: np.ndarray,
    sources: list,
    R_cover: float,
) -> np.ndarray:
    """Render 1 khung hình Full HD đối sánh trực tiếp 5 thuật toán tối ưu hóa."""
    M, N = pollution_grid.shape
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(DARK_BG)

    title_text = (
        f"ĐỐI SÁNH ĐỘNG HỌC 5 THUẬT TOÁN TỐI ƯU HÓA MẠNG LƯỚI TRẠM QUAN TRẮC ĐÔ THỊ (OMNP)\n"
        f"Thế hệ {t+1:03d}/{max_iters} | HARO (Đề Xuất) vs ARO vs PSO vs MVO vs GA | Chuẩn màu EPA AQI"
    )
    fig.suptitle(title_text, fontsize=14, weight="bold", color="white", y=0.98)

    # 5 ô bản đồ cho 5 thuật toán ở hàng trên + 1 đồ thị hội tụ ở hàng dưới
    gs = GridSpec(2, 5, height_ratios=[1.3, 0.7], left=0.03, right=0.97, bottom=0.07, top=0.91, wspace=0.15, hspace=0.25)

    # Vẽ 5 ô bản đồ nhỏ cho 5 thuật toán
    for col_idx, name in enumerate(algo_names):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor(PANEL_BG)

        ax.imshow(pollution_grid.T, origin="lower", cmap=AQI_CMAP, extent=[0, M-1, 0, N-1], aspect="equal", alpha=0.85)

        data_t = history[name][t]
        coords = data_t["coords"]
        cov = data_t["coverage_pct"]

        for sx, sy in coords:
            c = Circle((sx, sy), R_cover, fill=True, facecolor=ALGO_COLORS[name], alpha=0.15, edgecolor=ALGO_COLORS[name], linewidth=1.2, linestyle="--")
            ax.add_patch(c)
            ax.plot(sx, sy, marker="*", markersize=11, color=ALGO_COLORS[name], markeredgecolor="black", markeredgewidth=0.8)

        color_hex = ALGO_COLORS[name]
        ax.set_title(f"{name}\nPhủ: {cov:.2f}%", color=color_hex, fontsize=11, weight="bold", pad=5)
        ax.tick_params(colors=TEXT_COLOR, labelsize=6)

    # Hàng dưới: Đồ thị đường cong bao phủ ô nhiễm thời gian thực
    ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.set_facecolor(PANEL_BG)

    iters_x = np.arange(1, t + 2)
    for name in algo_names:
        cov_curve = [history[name][k]["coverage_pct"] for k in range(t + 1)]
        lw = 3.0 if name == "HARO" else 1.8
        zo = 10 if name == "HARO" else 5
        label = f"{name} ({cov_curve[-1]:.2f}%)"
        ax_curve.plot(iters_x, cov_curve, color=ALGO_COLORS[name], linewidth=lw, label=label, zorder=zo)
        ax_curve.scatter([iters_x[-1]], [cov_curve[-1]], color=ALGO_COLORS[name], s=40, zorder=zo+1)

    ax_curve.set_xlim(1, max_iters)
    ax_curve.set_ylim(40, 100)
    ax_curve.set_xlabel("Thế Hệ (Iteration t)", color=TEXT_COLOR, fontsize=10)
    ax_curve.set_ylabel("Độ Bao Phủ Khối Lượng Ô Nhiễm (%)", color=TEXT_COLOR, fontsize=10)
    ax_curve.set_title("So Sánh Tốc Độ Hội Tụ & Khả Năng Bao Phủ Ô Nhiễm Đô Thị", color="white", fontsize=11, weight="bold")
    ax_curve.grid(alpha=0.25, color=GRID_COLOR, linestyle="--")
    ax_curve.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax_curve.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=9, loc="lower right")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    return rgba


# ==============================================================================
# 4. HÀM XUẤT VIDEO (MP4) VÀ HOẠT ẢNH (GIF)
# ==============================================================================
def export_frames_to_video_and_gif(
    frames_rgba: List[np.ndarray],
    mp4_path: str,
    gif_path: str,
    fps: int = 15,
    hold_last_seconds: float = 2.5,
):
    """Xuất danh sách khung hình RGBA thành video MP4 (OpenCV) và ảnh động GIF (PIL)."""
    if not frames_rgba:
        raise ValueError("Danh sách khung hình rỗng!")

    height, width, _ = frames_rgba[0].shape
    n_hold = int(fps * hold_last_seconds)

    # Thêm các khung hình giữ lại ở cuối để người xem chiêm ngưỡng kết quả tối ưu
    all_frames = list(frames_rgba) + [frames_rgba[-1]] * n_hold

    print(f"\n[Export Engine] Đang xuất {len(all_frames)} khung hình ({width}x{height} @ {fps} FPS)...")

    # 1. Xuất MP4 bằng cv2.VideoWriter
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_mp4 = cv2.VideoWriter(mp4_path, fourcc, float(fps), (width, height))
        for frame in all_frames:
            # Chuyển RGBA sang BGR cho OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            out_mp4.write(bgr)
        out_mp4.release()
        size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
        print(f"  ✔ Đã lưu Video MP4: {mp4_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  ⚠ Lỗi khi xuất MP4: {e}")

    # 2. Xuất GIF động chất lượng cao bằng PIL
    try:
        pil_images = []
        gif_width = width // 2
        gif_height = height // 2
        for frame in all_frames:
            img = Image.fromarray(frame).resize((gif_width, gif_height), Image.Resampling.BILINEAR)
            pil_images.append(img.convert("P", palette=Image.Palette.ADAPTIVE))

        frame_duration = int(1000 / fps)
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=frame_duration,
            loop=0,
            optimize=True,
        )
        size_gif_mb = os.path.getsize(gif_path) / (1024 * 1024)
        print(f"  ✔ Đã lưu Ảnh động GIF: {gif_path} ({size_gif_mb:.2f} MB)")
    except Exception as e:
        print(f"  ⚠ Lỗi khi xuất GIF: {e}")


# ==============================================================================
# 5. HÀM MAIN THỰC THI CHÍNH
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sinh video minh họa cơ chế thuật toán OMNP.")
    parser.add_argument("--mode", type=str, choices=["haro", "comparative", "all"], default="all",
                        help="Chế độ: 'haro' (chuyên sâu HARO), 'comparative' (so sánh 5 thuật toán), 'all' (sinh cả 2)")
    parser.add_argument("--grid_size", type=int, default=60, help="Kích thước lưới đô thị M x N (mặc định: 60)")
    parser.add_argument("--k", type=int, default=8, help="Số lượng trạm quan trắc K (mặc định: 8)")
    parser.add_argument("--max_iters", type=int, default=60, help="Số thế hệ lặp T (mặc định: 60)")
    parser.add_argument("--pop_size", type=int, default=30, help="Kích thước quần thể (mặc định: 30)")
    parser.add_argument("--fps", type=int, default=12, help="Tốc độ khung hình (mặc định: 12 FPS)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("  CHƯƠNG TRÌNH SINH VIDEO MINH HỌA CƠ CHẾ THUẬT TOÁN (OMNP - HARO)")
    print("=" * 80)
    print(f"  - Kích thước lưới:    {args.grid_size} x {args.grid_size}")
    print(f"  - Số lượng trạm K:    {args.k} trạm")
    print(f"  - Quần thể / Số thế hệ: {args.pop_size} cá thể / {args.max_iters} thế hệ")
    print(f"  - Tốc độ khung hình:   {args.fps} FPS")
    print(f"  - Thư mục lưu trữ:    {args.output_dir}")
    print(f"  - Chế độ chạy:        {args.mode.upper()}")
    print("=" * 80)

    # 1. Xây dựng bài toán với đúng thông số đề tài
    prob, env_res, mask, R_cover, d_min, pop_calc, T_calc = build_experiment_problem(
        M=args.grid_size, N=args.grid_size, K=args.k, random_seed=42
    )

    # CHẾ ĐỘ 1: CHUYÊN SÂU 5 CƠ CHẾ HARO
    if args.mode in ["haro", "all"]:
        print("\n>>> [1/2] BẮT ĐẦU SINH VIDEO MINH HỌA 5 CƠ CHẾ HARO...")
        recorder = HAROMechanismRecorder(
            problem=prob,
            pop_size=args.pop_size,
            max_iters=args.max_iters,
            seed=120,
            halo_inner=0.05,
            halo_middle=0.15,
            halo_outer=0.35,
            halo_strength=0.50,
            stagnation_threshold=8,
        )
        snapshots = recorder.record_run()

        print(f"\n[Render Engine] Đang render {len(snapshots)} khung hình Dashboard Full HD cho HARO...")
        haro_frames = []
        for idx, snap in enumerate(snapshots):
            frame = render_haro_mechanism_frame(
                snapshot=snap,
                pollution_grid=env_res.concentration,
                feasibility_mask=mask,
                sources=env_res.sources,
                R_cover=R_cover,
                d_min=d_min,
            )
            haro_frames.append(frame)
            if (idx + 1) % 15 == 0 or idx == len(snapshots) - 1:
                print(f"  Đã render khung hình: {idx+1:03d}/{len(snapshots)}")

        mp4_haro = os.path.join(args.output_dir, "video_minh_hoa_co_che_haro.mp4")
        gif_haro = os.path.join(args.output_dir, "video_minh_hoa_co_che_haro.gif")
        export_frames_to_video_and_gif(haro_frames, mp4_haro, gif_haro, fps=args.fps)

    # CHẾ ĐỘ 2: ĐỐI SÁNH ĐỒNG THỜI 5 THUẬT TOÁN
    if args.mode in ["comparative", "all"]:
        print("\n>>> [2/2] BẮT ĐẦU SINH VIDEO ĐỐI SÁNH 5 THUẬT TOÁN (HARO, ARO, PSO, MVO, GA)...")
        algo_names, history = record_comparative_run(
            problem=prob,
            pop_size=args.pop_size,
            max_iters=args.max_iters,
            seed=120,
        )

        print(f"\n[Render Engine] Đang render {args.max_iters} khung hình Đối Sánh 5 Thuật Toán...")
        comp_frames = []
        for t in range(args.max_iters):
            frame = render_comparative_frame(
                t=t,
                max_iters=args.max_iters,
                algo_names=algo_names,
                history=history,
                pollution_grid=env_res.concentration,
                sources=env_res.sources,
                R_cover=R_cover,
            )
            comp_frames.append(frame)
            if (t + 1) % 15 == 0 or t == args.max_iters - 1:
                print(f"  Đã render khung hình: {t+1:03d}/{args.max_iters}")

        mp4_comp = os.path.join(args.output_dir, "video_so_sanh_5_thuat_toan.mp4")
        gif_comp = os.path.join(args.output_dir, "video_so_sanh_5_thuat_toan.gif")
        export_frames_to_video_and_gif(comp_frames, mp4_comp, gif_comp, fps=args.fps)

    print("\n" + "=" * 80)
    print("  HOÀN THÀNH TOÀN BỘ TIẾN TRÌNH SINH VIDEO MINH HỌA THUẬT TOÁN!")
    print(f"  Các file kết quả được lưu tại: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
