from __future__ import annotations

from typing import Optional, List, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, Patch


# ====================================================================== #
# BẢNG MÀU AQI CHUẨN EPA: xanh lá -> vàng -> cam -> đỏ -> tím
# ====================================================================== #
AQI_CMAP = LinearSegmentedColormap.from_list(
    "aqi_scale",
    [
        (0.00, "#00A65A"),  # xanh lá đậm  - nồng độ rất thấp (Tốt)
        (0.20, "#7CD100"),  # xanh lá nhạt - Tốt/Trung bình
        (0.40, "#FFF200"),  # vàng         - Trung bình
        (0.60, "#FF8C00"),  # cam          - Kém (nhạy cảm)
        (0.80, "#E8291C"),  # đỏ           - Xấu
        (1.00, "#7A1FA2"),  # tím          - Rất xấu / cực cao
    ],
    N=256,
)

DARK_BG = "#0e1117"
PANEL_BG = "#12161c"
GRID_COLOR = "#2a2f3a"
TEXT_COLOR = "#eaeaea"

# Classification marker and color style for emission sources
CATEGORY_STYLE = {
    "traffic":      dict(marker="o", color="white",   label="Traffic"),
    "Traffic":      dict(marker="o", color="white",   label="Traffic"),
    "Giao thông":   dict(marker="o", color="white",   label="Traffic"),
    "industrial":   dict(marker="s", color="#5ad1ff", label="Industrial"),
    "Industrial":   dict(marker="s", color="#5ad1ff", label="Industrial"),
    "Công nghiệp":  dict(marker="s", color="#5ad1ff", label="Industrial"),
    "construction": dict(marker="^", color="#ffd23f", label="Construction"),
    "Construction": dict(marker="^", color="#ffd23f", label="Construction"),
    "Xây dựng":     dict(marker="^", color="#ffd23f", label="Construction"),
    "noise":        dict(marker="D", color="#c9c9c9", label="Background / Noise"),
    "Noise":        dict(marker="D", color="#c9c9c9", label="Background / Noise"),
    "Background / Noise": dict(marker="D", color="#c9c9c9", label="Background / Noise"),
    "Nhiễu môi trường": dict(marker="D", color="#c9c9c9", label="Background / Noise"),
}

# Distinct palette for metaheuristic optimizers
ALGO_COLORS = {
    "HARO": "#39ff14",  # Neon Green (Proposed Algorithm)
    "ARO":  "#00e5ff",  # Cyan Blue (Baseline ARO)
    "PSO":  "#ff007f",  # Magenta (Particle Swarm Optimization)
    "MVO":  "#ffd600",  # Gold (Multi-Verse Optimizer)
    "GA":   "#b388ff",  # Lavender (Genetic Algorithm)
}


def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)


def _draw_sources(ax, sources):
    """Draw classified emission source symbols on the axis."""
    if not sources:
        return
    seen = set()
    for s in sources:
        cat = getattr(s, "source_type", getattr(s, "category", "Traffic"))
        if hasattr(cat, "value"):
            cat = cat.value
        style = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["Traffic"])
        ax.plot(
            s.x, s.y, marker=style["marker"], markersize=9,
            markerfacecolor=style["color"], markeredgecolor="black",
            markeredgewidth=0.8, linestyle="none", zorder=6,
        )
        seen.add(style["label"])

    handles = [
        plt.Line2D([0], [0], marker=s_dict["marker"], color="none",
                   markerfacecolor=s_dict["color"], markeredgecolor="black",
                   markersize=8, label=label)
        for label, s_dict in [("Traffic", CATEGORY_STYLE["Traffic"]),
                              ("Industrial", CATEGORY_STYLE["Industrial"]),
                              ("Construction", CATEGORY_STYLE["Construction"]),
                              ("Background / Noise", CATEGORY_STYLE["Background / Noise"])]
        if label in seen
    ]
    if handles:
        leg = ax.legend(handles=handles, loc="upper right", framealpha=0.85,
                         facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=8)
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)


def compute_coverage_metrics(
    pollution_grid: np.ndarray,
    stations: np.ndarray,
    R_cover: float,
) -> tuple[float, float]:
    """Tính tổng nồng độ ô nhiễm được bao phủ F và tỉ lệ % bao phủ."""
    M, N = pollution_grid.shape
    xs = np.arange(M)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    covered_mask = np.zeros((M, N), dtype=bool)
    for (sx, sy) in stations:
        dist_sq = (XX - sx) ** 2 + (YY - sy) ** 2
        covered_mask |= (dist_sq <= R_cover ** 2)

    F = float(pollution_grid[covered_mask].sum())
    total = float(pollution_grid.sum())
    coverage_pct = 100.0 * F / total if total > 0 else 0.0
    return F, coverage_pct


def plot_heatmap_with_stations(
    pollution_grid: np.ndarray,
    stations: np.ndarray,
    R_cover: float,
    feasibility_mask: Optional[np.ndarray] = None,
    sources: Optional[list] = None,
    F: Optional[float] = None,
    coverage_pct: Optional[float] = None,
    title: str = "Heatmap + Trạm Quan Trắc",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Vẽ bản đồ nhiệt nồng độ ô nhiễm (chuẩn AQI), các vùng cấm (Restricted Zones),
    vị trí K trạm quan trắc tối ưu (sao sáng) và vòng tròn bao phủ R_cover."""
    if F is None or coverage_pct is None:
        F_calc, cov_calc = compute_coverage_metrics(pollution_grid, stations, R_cover)
        if F is None:
            F = F_calc
        if coverage_pct is None:
            coverage_pct = cov_calc

    M, N = pollution_grid.shape
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    _apply_dark_style(ax, fig)

    im = ax.imshow(
        pollution_grid.T,
        origin="lower",
        cmap=AQI_CMAP,
        extent=[0, M - 1, 0, N - 1],
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Air Quality Index (AQI) / Concentration C(x, y)", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)

    legend_handles = []

    # Draw restricted areas from feasibility_mask (value 0 represents restricted zone)
    if feasibility_mask is not None:
        for x in range(M):
            for y in range(N):
                if feasibility_mask[x, y] == 0:
                    rect = Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        facecolor="#ff3b3b", alpha=0.30, edgecolor="none", zorder=2,
                    )
                    ax.add_patch(rect)
        legend_handles.append(Patch(color="#ff3b3b", alpha=0.30, label="Restricted Zone"))

    # Draw emission sources
    if sources:
        seen = set()
        for s in sources:
            cat = getattr(s, "source_type", getattr(s, "category", "Traffic"))
            if hasattr(cat, "value"):
                cat = cat.value
            style = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["Traffic"])
            ax.plot(
                s.x, s.y, marker=style["marker"], markersize=9,
                markerfacecolor=style["color"], markeredgecolor="black",
                markeredgewidth=0.8, linestyle="none", zorder=6,
            )
            seen.add(style["label"])
        for label, s_dict in [("Traffic", CATEGORY_STYLE["Traffic"]),
                              ("Industrial", CATEGORY_STYLE["Industrial"]),
                              ("Construction", CATEGORY_STYLE["Construction"]),
                              ("Background / Noise", CATEGORY_STYLE["Background / Noise"])]:
            if label in seen:
                legend_handles.append(
                    plt.Line2D([0], [0], marker=s_dict["marker"], color="none",
                               markerfacecolor=s_dict["color"], markeredgecolor="black",
                               markersize=8, label=label)
                )

    # Draw optimal monitoring stations
    for idx, (sx, sy) in enumerate(stations):
        ax.plot(sx, sy, marker="*", markersize=17, color="#00ffe1",
                markeredgecolor="black", markeredgewidth=1.0, zorder=7)
        ax.annotate(f"S{idx+1:02d}", (sx, sy), textcoords="offset points",
                    xytext=(6, 8), color="white", fontsize=9, weight="bold", zorder=8)
        circle = Circle(
            (sx, sy), R_cover, fill=False, edgecolor="white",
            linewidth=1.6, linestyle="--", alpha=0.9, zorder=5,
        )
        ax.add_patch(circle)

    legend_handles.append(
        plt.Line2D([0], [0], color="white", linestyle="--", linewidth=1.6,
                   label=f"Sensing Radius (Rc={R_cover:g})")
    )
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#00ffe1",
                   markeredgecolor="black", markersize=12, label="Monitoring Station")
    )

    leg = ax.legend(handles=legend_handles, loc="upper right", framealpha=0.85,
                    facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color("white")

    ax.set_xlim(0, M - 1)
    ax.set_ylim(0, N - 1)
    ax.set_xlabel("X-coordinate (Grid Units)")
    ax.set_ylabel("Y-coordinate (Grid Units)")

    full_title = f"{title} | F = {F:,.0f} | Coverage: {coverage_pct:.2f}%"
    ax.set_title(full_title, fontsize=12, weight="bold", color="white")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_pollution_dashboard(
    pollution_grid: np.ndarray,
    sources: Optional[list] = None,
    title: str = "Urban Air Quality Simulation Dashboard",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Two-panel publication dashboard:
      (Left)  Pollution Heatmap + Categorized Emission Sources
      (Right) Isoconcentration Contour Map
    """
    M, N = pollution_grid.shape
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(title, fontsize=15, weight="bold", color=TEXT_COLOR)

    # Left panel: Heatmap
    ax1 = axes[0]
    _apply_dark_style(ax1, fig)
    im1 = ax1.imshow(pollution_grid.T, origin="lower", cmap=AQI_CMAP,
                     extent=[0, M - 1, 0, N - 1], aspect="equal")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Air Quality Index (AQI 0-100)", color=TEXT_COLOR)
    cbar1.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar1.ax.axes, "yticklabels"), color=TEXT_COLOR)
    ax1.set_title("Pollution Heatmap & Emission Sources", color=TEXT_COLOR)
    ax1.set_xlabel("X-coordinate (Grid Units)")
    ax1.set_ylabel("Y-coordinate (Grid Units)")
    _draw_sources(ax1, sources)

    # Right panel: Contour map
    ax2 = axes[1]
    _apply_dark_style(ax2, fig)
    im2 = ax2.imshow(pollution_grid.T, origin="lower", cmap=AQI_CMAP,
                     extent=[0, M - 1, 0, N - 1], aspect="equal", alpha=0.95)
    xs = np.arange(M)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    cs = ax2.contour(XX, YY, pollution_grid, levels=6, colors="white",
                     linewidths=0.9, alpha=0.85)
    ax2.clabel(cs, inline=True, fontsize=7, colors="white")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Air Quality Index (AQI 0-100)", color=TEXT_COLOR)
    cbar2.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color=TEXT_COLOR)
    ax2.set_title("Isoconcentration Contour Map", color=TEXT_COLOR)
    ax2.set_xlabel("X-coordinate (Grid Units)")
    ax2.set_ylabel("Y-coordinate (Grid Units)")
    _draw_sources(ax2, sources)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig


def plot_convergence_comparison(
    algo_runs: Dict[str, List[np.ndarray]],
    title: str = "Convergence Characteristics of Optimization Algorithms",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot convergence curves comparing all 5 algorithms (HARO, ARO, PSO, MVO, GA).

    Displays:
      - Solid curve for Mean fitness across 30 independent runs.
      - Translucent shaded ribbon for Standard Deviation band (Mean ± STD).
      - Final Best and STD metrics annotated in the Legend.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    _apply_dark_style(ax, fig)

    for algo_name, runs in algo_runs.items():
        if not runs:
            continue
        arr = np.array(runs)  # shape (n_runs, n_iters)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        final_best = float(np.min(arr[:, -1]))
        final_mean = float(mean_curve[-1])
        final_std = float(std_curve[-1])

        color = ALGO_COLORS.get(algo_name, "#ffffff")
        lw = 2.4 if algo_name == "HARO" else 1.6
        zorder = 10 if algo_name == "HARO" else 5

        iters = np.arange(1, len(mean_curve) + 1)
        label = f"{algo_name} (Best: {final_best:.4f} | Std: {final_std:.4f})"

        # Plot Mean curve
        ax.plot(iters, mean_curve, color=color, linewidth=lw, label=label, zorder=zorder)

        # Plot STD band
        ax.fill_between(
            iters,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.15,
            zorder=zorder - 1,
        )

    ax.set_xlabel("Iteration (Generation t)", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Fitness Function f(S) [Minimization]", fontsize=11, color=TEXT_COLOR)
    ax.set_title(title, fontsize=13, weight="bold", color="white", pad=12)
    ax.grid(alpha=0.25, color=GRID_COLOR, linestyle="--")

    leg = ax.legend(
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        fontsize=9,
        loc="upper right",
        framealpha=0.9,
    )
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    return fig
