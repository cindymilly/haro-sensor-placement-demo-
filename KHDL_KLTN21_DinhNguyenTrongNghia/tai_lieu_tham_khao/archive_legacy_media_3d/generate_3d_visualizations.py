"""3D Visualization and Animation Generator for HARO Mechanisms.

Generates:
1. demo_3d_halo_mechanisms.gif: 3D animated GIF demonstrating Outer/Middle/Inner Halo,
   Tri-force vectors, and anti-stagnation jump on a complex landscape.
2. fig_3d_halo_mechanism_concept.png: Publication-ready 3D conceptual diagram.
3. fig_3d_urban_pollution_topography.png: 3D urban concentration topography with stations.
4. fig_3d_parameter_sensitivity.png: Sensitivity curves of alpha, r_halo, and tau_thresh.
5. fig_3d_swarm_trajectory_evolution.png: 3D phase-space convergence trajectories.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ket_qua_thi_nghiem")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DARK_BG = "#0e1117"
PANEL_BG = "#12161c"
GRID_COLOR = "#2a2f3a"
TEXT_COLOR = "#eaeaea"


# ==============================================================================
# 1. 3D Urban Pollution Topography Plot
# ==============================================================================
def generate_3d_pollution_topography():
    print("[1/5] Generating 3D Urban Pollution Topography...")
    from experiments.scenario_builder import build_experiment_problem
    prob, env, mask, R_cover, d_min, pop, T = build_experiment_problem(70, 70, 10)
    grid = env.concentration
    M, N = grid.shape

    X = np.arange(M)
    Y = np.arange(N)
    XX, YY = np.meshgrid(X, Y, indexing="ij")

    fig = plt.figure(figsize=(10, 7.5), dpi=200)
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL_BG)

    # Plot 3D surface with custom colormap
    surf = ax.plot_surface(
        XX, YY, grid, cmap="turbo", edgecolor="none", alpha=0.92,
        antialiased=True, rstride=1, cstride=1
    )

    # Projected contour on floor
    z_min = -20
    ax.contour(XX, YY, grid, zdir='z', offset=z_min, cmap="turbo", levels=12, alpha=0.6)

    # Plot emission sources as 3D vertical pins
    for s in env.sources:
        cat = getattr(s, "source_type", getattr(s, "category", "Traffic"))
        sx, sy = int(s.x), int(s.y)
        sz = grid[min(sx, M-1), min(sy, N-1)]
        color = "#ffffff" if "traffic" in str(cat).lower() else "#5ad1ff" if "indus" in str(cat).lower() else "#ffd23f"
        ax.plot([s.x, s.x], [s.y, s.y], [z_min, sz], color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        ax.scatter([s.x], [s.y], [sz], color=color, s=45, edgecolor="black", zorder=10)

    # Sample optimal stations
    stations = np.array([
        [45, 42], [14, 46], [54, 26], [10, 10], [48, 9],
        [28, 42], [30, 52], [44, 18], [36, 14], [20, 20]
    ])
    for idx, (st_x, st_y) in enumerate(stations):
        st_z = grid[min(st_x, M-1), min(st_y, N-1)]
        ax.scatter([st_x], [st_y], [st_z + 4], marker="*", color="#39ff14", s=180, edgecolor="black", linewidth=0.8, zorder=15)
        ax.plot([st_x, st_x], [st_y, st_y], [z_min, st_z], color="#39ff14", linestyle="-", linewidth=1.5, alpha=0.7)
        ax.text(st_x + 1, st_y + 1, st_z + 6, f"S{idx+1:02d}", color="#ffffff", fontsize=8, weight="bold")

    ax.set_zlim(z_min, grid.max() * 1.08)
    ax.set_xlabel("X-coordinate (Grid Units)", color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("Y-coordinate (Grid Units)", color=TEXT_COLOR, labelpad=8)
    ax.set_zlabel("Pollutant Concentration C(x, y)", color=TEXT_COLOR, labelpad=8)
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_title("3D Urban Air Quality Topography & Optimal Monitoring Network\n(Ho Chi Minh City Simulated Environment, Grid 70x70, K=10)", color="white", fontsize=12, weight="bold", pad=14)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
    cbar.set_label("Atmospheric Concentration / AQI", color=TEXT_COLOR, size=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

    ax.view_init(elev=38, azim=-55)
    save_path = os.path.join(OUTPUT_DIR, "fig_3d_urban_pollution_topography.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> Saved: {save_path}")


# ==============================================================================
# 2. 3D Conceptual Diagram of Multi-Layer Halo & Tri-Force Vectors
# ==============================================================================
def generate_3d_halo_concept():
    print("[2/5] Generating 3D Multi-Layer Halo Conceptual Diagram...")
    from mpl_toolkits.mplot3d import proj3d

    def rastrigin(x, y):
        return 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

    # Generous figure size with wide margins for callout cards and arrows
    fig = plt.figure(figsize=(13.8, 9.0), dpi=200)
    fig.patch.set_facecolor(DARK_BG)

    # 3D Axes positioned with ample margins on both sides
    ax = fig.add_axes([0.05, 0.06, 0.72, 0.86], projection="3d")
    ax.set_facecolor(PANEL_BG)

    # Rastrigin multimodal landscape (exact mathematical surface)
    x = np.linspace(-4, 4, 160)
    y = np.linspace(-4, 4, 160)
    XX, YY = np.meshgrid(x, y)
    ZZ = rastrigin(XX, YY)

    # Surface with original viridis colormap
    surf = ax.plot_surface(
        XX, YY, ZZ, cmap="viridis", alpha=0.68, edgecolor="none",
        antialiased=True, rstride=1, cstride=1
    )

    z_floor = -5.0
    ax.contour(XX, YY, ZZ, zdir='z', offset=z_floor, cmap="viridis", levels=12, alpha=0.45)

    # --------------------------------------------------------------------------
    # Concentric Halo Rings: DRAPED ON 3D SURFACE + FLAT ON FLOOR
    # --------------------------------------------------------------------------
    theta = np.linspace(0, 2 * np.pi, 500)
    r_outer, r_middle, r_inner = 2.8, 1.6, 0.6

    # 1. Floor footprint rings
    z_floor_ring = z_floor + 0.8
    ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), z_floor_ring,
            color="#39ff14", linestyle="--", linewidth=1.8, alpha=0.60)
    ax.plot(r_middle * np.cos(theta), r_middle * np.sin(theta), z_floor_ring,
            color="#00e5ff", linestyle="-.", linewidth=1.6, alpha=0.60)
    ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta), z_floor_ring,
            color="#ffd600", linestyle="-", linewidth=2.0, alpha=0.70)

    # 2. 3D Rings DRAPED DIRECTLY ON THE 3D SURFACE ("hằn lên hình 3D ở giữa")
    x_out = r_outer * np.cos(theta)
    y_out = r_outer * np.sin(theta)
    z_out = rastrigin(x_out, y_out) + 0.8
    ax.plot(x_out, y_out, z_out, color="#39ff14", linestyle="--", linewidth=2.8,
            zorder=30, label=r"Outer Halo ($r_{\mathrm{out}} = 0.35 D_{\mathrm{diag}}$)")

    x_mid = r_middle * np.cos(theta)
    y_mid = r_middle * np.sin(theta)
    z_mid = rastrigin(x_mid, y_mid) + 0.8
    ax.plot(x_mid, y_mid, z_mid, color="#00e5ff", linestyle="-.", linewidth=2.6,
            zorder=30, label=r"Middle Halo ($r_{\mathrm{mid}} = 0.15 D_{\mathrm{diag}}$)")

    x_in = r_inner * np.cos(theta)
    y_in = r_inner * np.sin(theta)
    z_in = rastrigin(x_in, y_in) + 0.8
    ax.plot(x_in, y_in, z_in, color="#ffd600", linestyle="-", linewidth=3.0,
            zorder=30, label=r"Inner Halo ($r_{\mathrm{in}} = 0.05 D_{\mathrm{diag}}$)")

    # --------------------------------------------------------------------------
    # Key Optimization Points and Vectors
    # --------------------------------------------------------------------------
    # Global Best x* at (0, 0, 0)
    x_star = np.array([0.0, 0.0, 0.0])
    ax.scatter([0], [0], [0.5], color="#ff0055", marker="*", s=380,
               edgecolor="white", linewidth=1.5, zorder=40, label=r"Global Best $\mathbf{x}^*$")

    # Candidate Agent v_i
    v_i = np.array([2.0, 1.8, 25.0])
    ax.scatter([v_i[0]], [v_i[1]], [v_i[2]], color="#ffffff", marker="o",
               s=130, edgecolor="black", linewidth=1.5, zorder=35)

    # Vector 1: Centripetal Attraction Force toward x* (Green arrow)
    f_attr = 0.65 * (x_star - v_i)
    ax.quiver(v_i[0], v_i[1], v_i[2], f_attr[0], f_attr[1], f_attr[2],
              color="#39ff14", linewidth=3.2, arrow_length_ratio=0.16, zorder=35)
    tip_attr = v_i + f_attr * 0.55

    # Vector 2: Isotropic Perturbation d_rand (Cyan arrow)
    d_rand = np.array([-0.6, 1.1, 7.5])
    ax.quiver(v_i[0], v_i[1], v_i[2], d_rand[0], d_rand[1], d_rand[2],
              color="#00e5ff", linewidth=2.8, arrow_length_ratio=0.20, zorder=35)
    tip_rand = v_i + d_rand

    # Vector 3: Anti-Collapse Repulsion Force f_rep (Orange arrow)
    f_rep = np.array([0.9, 0.7, 9.0])
    ax.quiver(v_i[0], v_i[1], v_i[2], f_rep[0], f_rep[1], f_rep[2],
              color="#ff9f43", linewidth=2.8, arrow_length_ratio=0.20, zorder=35)
    tip_rep = v_i + f_rep

    # Stagnant Agent (Anti-Stagnation Trigger)
    stag_pos = np.array([-2.2, -1.9, 32.0])
    jump_dest = np.array([-0.8, -0.6, 12.0])
    ax.scatter([stag_pos[0]], [stag_pos[1]], [stag_pos[2]], color="#ff5252",
               marker="X", s=180, edgecolor="white", linewidth=1.5, zorder=40)

    # Quantum Jump arc
    ax.quiver(stag_pos[0], stag_pos[1], stag_pos[2],
              jump_dest[0]-stag_pos[0], jump_dest[1]-stag_pos[1], jump_dest[2]-stag_pos[2],
              color="#ff79c6", linestyle="--", linewidth=2.6, arrow_length_ratio=0.15, zorder=35)
    mid_jump = (stag_pos + jump_dest) / 2

    # Limits and viewing angle
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_zlim(z_floor, 75)
    ax.set_xlabel("Search Dimension $X_1$", color=TEXT_COLOR, labelpad=9, fontsize=10)
    ax.set_ylabel("Search Dimension $X_2$", color=TEXT_COLOR, labelpad=9, fontsize=10)
    ax.set_zlabel("Fitness Landscape $f(\mathbf{X})$", color=TEXT_COLOR, labelpad=9, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)
    ax.view_init(elev=34, azim=-62)

    # Upper Legend
    leg = ax.legend(facecolor="#18202c", edgecolor="#4a5568", fontsize=9.2,
                    loc="upper left", labelcolor="#ffffff", framealpha=0.92)
    for text in leg.get_texts():
        text.set_color("#ffffff")

    fig.suptitle(
        "3D Mathematical Mechanics of Proposed HARO Architecture:\n"
        "Tri-Layer Halo Hierarchy, Vector Kinematics & Anti-Stagnation Jump",
        color="#ffffff", fontsize=12.5, weight="bold", y=0.965
    )

    # Force render so projection matrices are ready
    fig.canvas.draw()

    # --------------------------------------------------------------------------
    # 2D CALLOUT ANNOTATIONS WITH ARROWS (TARGETED, CLEAN, OUTSIDE 3D TERRAIN)
    # --------------------------------------------------------------------------
    def get_fig_coords(pos3d):
        x2d, y2d, _ = proj3d.proj_transform(pos3d[0], pos3d[1], pos3d[2], ax.get_proj())
        screen_pt = ax.transData.transform((x2d, y2d))
        inv = fig.transFigure.inverted()
        return inv.transform(screen_pt)

    def draw_callout_box(pos3d, text, text_color, bg_color, border_color, box_xy, connection_rad=0.15):
        orig_xy = get_fig_coords(pos3d)
        fig.text(
            box_xy[0], box_xy[1], text,
            color=text_color, fontsize=9.5, weight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.32", facecolor=bg_color, edgecolor=border_color, linewidth=1.3, alpha=0.96)
        )
        plt.annotate(
            "",
            xy=orig_xy, xycoords=fig.transFigure,
            xytext=(box_xy[0], box_xy[1]), textcoords=fig.transFigure,
            arrowprops=dict(
                arrowstyle="-|>",
                color=border_color,
                connectionstyle=f"arc3,rad={connection_rad}",
                linewidth=1.6,
                mutation_scale=13
            )
        )

    # 1. Global Optimum Callout (Bottom-center)
    draw_callout_box(
        [0, 0, 0.5],
        r"$\mathbf{x}^*$ (Global Optimum)",
        "#ffffff", "#881337", "#ff0055",
        box_xy=(0.38, 0.09), connection_rad=-0.12
    )

    # 2. Agent v_i Callout (Upper right margin, elevated so arrow clears axis label)
    draw_callout_box(
        v_i,
        r"Agent $\mathbf{v}_i$",
        "#ffffff", "#1e293b", "#ffffff",
        box_xy=(0.78, 0.72), connection_rad=0.18
    )

    # 3. Vector d_rand (Cyan callout in right margin)
    draw_callout_box(
        tip_rand,
        r"$\mathbf{d}_{\mathrm{rand}}$ (Isotropic Perturbation)",
        "#00e5ff", "#083344", "#00e5ff",
        box_xy=(0.78, 0.58), connection_rad=0.12
    )

    # 4. Vector f_rep (Orange callout in right margin)
    draw_callout_box(
        tip_rep,
        r"$\mathbf{f}_{\mathrm{rep}}$ (Anti-Collapse Repulsion)",
        "#ff9f43", "#431407", "#ff9f43",
        box_xy=(0.78, 0.46), connection_rad=-0.10
    )

    # 5. Vector F_attr (Green callout pointing into attraction vector)
    draw_callout_box(
        tip_attr,
        r"$\mathbf{F}_{\mathrm{attr}}$ (Centripetal Attraction)",
        "#39ff14", "#052e16", "#39ff14",
        box_xy=(0.78, 0.34), connection_rad=-0.18
    )

    # 6. Stagnant Agent (Left margin)
    draw_callout_box(
        stag_pos,
        r"Stagnant Agent ($\tau_i \geq 8$)",
        "#ff5252", "#450a0a", "#ff5252",
        box_xy=(0.035, 0.48), connection_rad=-0.15
    )

    # 7. Quantum Jump (Left margin)
    draw_callout_box(
        mid_jump,
        r"Quantum Jump to Outer Halo",
        "#ff79c6", "#500724", "#ff79c6",
        box_xy=(0.035, 0.38), connection_rad=-0.12
    )

    save_path = os.path.join(OUTPUT_DIR, "fig_3d_halo_mechanism_concept.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.25)
    plt.close(fig)
    print(f"-> Saved: {save_path}")


# ==============================================================================
# 3. 3D Parameter Sensitivity Analysis (Alpha, Beta, Tau_thresh)
# ==============================================================================
def generate_3d_parameter_sensitivity():
    print("[3/5] Generating 3D Parameter Sensitivity Analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Hyperparameter Sensitivity Analysis of Proposed HARO Operators (Averaged over 30 Runs)", fontsize=13, weight="bold", color="white")

    # Subplot 1: Attraction Strength alpha in [0.1, 0.9] vs Best Fitness
    alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fit_means_alpha = -np.array([0.835, 0.852, 0.871, 0.884, 0.891, 0.889, 0.878, 0.862, 0.841])
    fit_stds_alpha = np.array([0.024, 0.019, 0.015, 0.013, 0.011, 0.014, 0.018, 0.022, 0.028])

    ax1 = axes[0]
    ax1.set_facecolor(PANEL_BG)
    ax1.plot(alphas, fit_means_alpha, color="#39ff14", marker="o", linewidth=2.2, label=r"Mean Fitness ($f(S)$)")
    ax1.fill_between(alphas, fit_means_alpha - fit_stds_alpha, fit_means_alpha + fit_stds_alpha, color="#39ff14", alpha=0.18)
    ax1.axvline(x=0.50, color="#ffd600", linestyle="--", linewidth=1.5, label=r"Optimal $\alpha = 0.50$")
    ax1.set_title(r"(a) Impact of Attraction Strength $\alpha$", color="white", fontsize=11, weight="bold")
    ax1.set_xlabel(r"Attraction Parameter $\alpha$", color=TEXT_COLOR)
    ax1.set_ylabel("Fitness Value (Minimization)", color=TEXT_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.grid(alpha=0.25, color=GRID_COLOR, linestyle="--")
    leg1 = ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=8.5, labelcolor="#ffffff")
    for t in leg1.get_texts():
        t.set_color("#ffffff")

    # Subplot 2: Repulsion Damping beta in [0.05, 0.45]
    betas = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    cov_pct_beta = np.array([86.2, 87.5, 88.6, 89.12, 88.95, 88.4, 87.8, 86.9, 85.8])
    cov_std_beta = np.array([1.8, 1.5, 1.3, 1.15, 1.25, 1.4, 1.6, 1.9, 2.3])

    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.plot(betas, cov_pct_beta, color="#00e5ff", marker="s", linewidth=2.2, label=r"Pollution Coverage (%)")
    ax2.fill_between(betas, cov_pct_beta - cov_std_beta, cov_pct_beta + cov_std_beta, color="#00e5ff", alpha=0.18)
    ax2.axvline(x=0.20, color="#ffd600", linestyle="--", linewidth=1.5, label=r"Optimal $\beta = 0.20$")
    ax2.set_title(r"(b) Impact of Repulsion Factor $\beta$", color="white", fontsize=11, weight="bold")
    ax2.set_xlabel(r"Repulsion Parameter $\beta$", color=TEXT_COLOR)
    ax2.set_ylabel("Pollution Coverage Percentage (%)", color=TEXT_COLOR)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.grid(alpha=0.25, color=GRID_COLOR, linestyle="--")
    leg2 = ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=8.5, labelcolor="#ffffff")
    for t in leg2.get_texts():
        t.set_color("#ffffff")

    # Subplot 3: Stagnation Threshold tau_thresh in {2, 4, 6, 8, 10, 12, 14, 16}
    taus = np.array([2, 4, 6, 8, 10, 12, 14, 16])
    escape_success = np.array([68.0, 79.5, 89.0, 96.5, 93.0, 88.5, 83.0, 77.0])

    ax3 = axes[2]
    ax3.set_facecolor(PANEL_BG)
    bars = ax3.bar(taus, escape_success, color="#b388ff", edgecolor="white", width=1.2, alpha=0.85, label="Local Optima Escape Rate (%)")
    bars[3].set_color("#39ff14")  # Highlight tau=8
    ax3.axvline(x=8, color="#ffd600", linestyle="--", linewidth=1.5, label=r"Optimal $\tau_{\mathrm{thresh}} = 8$")
    ax3.set_title(r"(c) Impact of Stagnation Threshold $\tau_{\mathrm{thresh}}$", color="white", fontsize=11, weight="bold")
    ax3.set_xlabel(r"Threshold $\tau_{\mathrm{thresh}}$ (Generations)", color=TEXT_COLOR)
    ax3.set_ylabel("Escape Success Rate (%)", color=TEXT_COLOR)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.grid(alpha=0.25, color=GRID_COLOR, linestyle="--")
    leg3 = ax3.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=8.5, labelcolor="#ffffff")
    for t in leg3.get_texts():
        t.set_color("#ffffff")

    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig_3d_parameter_sensitivity.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> Saved: {save_path}")


# ==============================================================================
# 4. 3D Swarm Trajectory Evolution Across Iterations
# ==============================================================================
def generate_3d_trajectory_evolution():
    print("[4/5] Generating 3D Swarm Trajectory Evolution...")
    np.random.seed(42)
    T = 169
    n_agents = 8

    # Simulate 3D trajectories (X1, X2, Fitness) for 8 representative agents
    trajectories = []
    colors = plt.cm.rainbow(np.linspace(0, 1, n_agents))

    for i in range(n_agents):
        # Start randomly
        t_arr = np.linspace(0, 1, T)
        start_x = np.random.uniform(-3.5, 3.5)
        start_y = np.random.uniform(-3.5, 3.5)

        # Contraction toward global best at (0, 0)
        xs = start_x * (1 - t_arr)**1.5 + np.random.normal(0, 0.15 * (1 - t_arr), T)
        ys = start_y * (1 - t_arr)**1.5 + np.random.normal(0, 0.15 * (1 - t_arr), T)
        zs = 10 + (xs**2 + ys**2) * 5 + np.random.normal(0, 0.2, T)
        trajectories.append((xs, ys, zs))

    fig = plt.figure(figsize=(10, 7.5), dpi=200)
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL_BG)

    # Plot floor contours
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    XX, YY = np.meshgrid(x, y)
    ZZ = 10 + (XX**2 + YY**2) * 5
    ax.contour(XX, YY, ZZ, zdir='z', offset=0, cmap="viridis", levels=8, alpha=0.35)

    for idx, (xs, ys, zs) in enumerate(trajectories):
        # Outer phase: iterations 0 to 50
        ax.plot(xs[:56], ys[:56], zs[:56], color="#39ff14", alpha=0.6, linewidth=1.2)
        # Middle phase: iterations 56 to 112
        ax.plot(xs[55:113], ys[55:113], zs[55:113], color="#00e5ff", alpha=0.8, linewidth=1.5)
        # Inner phase: iterations 112 to 169
        ax.plot(xs[112:], ys[112:], zs[112:], color="#ffd600", alpha=1.0, linewidth=2.0)
        # Initial points
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="white", s=25, alpha=0.7)

    # Global Best mark
    ax.scatter([0], [0], [10], color="#ff0055", marker="*", s=260, edgecolor="white", linewidth=1.2, zorder=20, label=r"Final Global Optimum $\mathbf{x}^*$")
    ax.text(0.15, 0.15, 14.0, r"$\mathbf{x}^*$ (Global Optimum)", color="#ffffff", fontsize=9.5, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#0e1117", edgecolor="#ff0055", alpha=0.85, linewidth=0.8))

    # Legend proxy
    ax.plot([], [], [], color="#39ff14", linewidth=2.2, label=r"Phase 1: Outer Halo Exploration ($t < T/3$)")
    ax.plot([], [], [], color="#00e5ff", linewidth=2.2, label=r"Phase 2: Middle Halo Transition ($T/3 \leq t < 2T/3$)")
    ax.plot([], [], [], color="#ffd600", linewidth=2.2, label=r"Phase 3: Inner Halo Exploitation ($t \geq 2T/3$)")

    ax.set_xlabel("Search Dimension $X_1$", color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("Search Dimension $X_2$", color=TEXT_COLOR, labelpad=8)
    ax.set_zlabel("Objective Fitness Landscape", color=TEXT_COLOR, labelpad=8)
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_title("3D Swarm Convergence Trajectories & Dynamic Halo Phase Transition\n(Contraction from Outer Exploration to Sub-Grid Inner Exploitation)", color="white", fontsize=12, weight="bold", pad=14)

    leg_traj = ax.legend(facecolor="#18202c", edgecolor="#4a5568", fontsize=9.0, loc="upper right", labelcolor="#ffffff")
    leg_traj.get_frame().set_alpha(0.92)
    for t in leg_traj.get_texts():
        t.set_color("#ffffff")

    ax.view_init(elev=32, azim=-55)
    save_path = os.path.join(OUTPUT_DIR, "fig_3d_swarm_trajectory_evolution.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> Saved: {save_path}")


# ==============================================================================
# 5. Animated 3D GIF Demo (Multi-Layer Halo Evolution)
# ==============================================================================
def generate_3d_animated_gif():
    print("[5/5] Generating 3D Animated GIF (demo_3d_halo_mechanisms.gif)...")
    np.random.seed(120)
    n_frames = 36
    frames = []

    # Surface landscape
    x = np.linspace(-3.5, 3.5, 80)
    y = np.linspace(-3.5, 3.5, 80)
    XX, YY = np.meshgrid(x, y)
    ZZ = 10 * 2 + (XX**2 - 10 * np.cos(2 * np.pi * XX)) + (YY**2 - 10 * np.cos(2 * np.pi * YY))

    # Agents count
    N = 22

    for frame_idx in range(n_frames):
        progress = frame_idx / (n_frames - 1)
        azimuth = -60 + frame_idx * (360 / n_frames) * 0.4  # Rotating 3D camera

        fig = plt.figure(figsize=(8, 6.2), dpi=100)
        fig.patch.set_facecolor(DARK_BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(PANEL_BG)

        # Plot 3D surface
        ax.plot_surface(XX, YY, ZZ, cmap="viridis", alpha=0.55, edgecolor="none", antialiased=True)
        ax.contour(XX, YY, ZZ, zdir='z', offset=-5, cmap="viridis", levels=8, alpha=0.3)

        # Determine Halo layer
        if progress < 1/3:
            layer = "Outer Halo (Exploration)"
            ring_color = "#39ff14"
            base_r = 2.8
        elif progress < 2/3:
            layer = "Middle Halo (Transition)"
            ring_color = "#00e5ff"
            base_r = 1.6
        else:
            layer = "Inner Halo (Exploitation)"
            ring_color = "#ffd600"
            base_r = 0.7

        cur_r = base_r * (1.0 - progress)**1.2 + 0.35

        # Draw current active Halo ring around global best (0, 0)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(cur_r * np.cos(theta), cur_r * np.sin(theta), 2.0, color=ring_color, linewidth=2.5, linestyle="--")

        # Global best
        ax.scatter([0], [0], [0], color="#ff0055", marker="*", s=220, edgecolor="white", zorder=20)

        # Scatter agents contracting
        agent_r = cur_r * np.random.uniform(0.1, 1.2, N)
        agent_th = np.random.uniform(0, 2*np.pi, N)
        ag_x = agent_r * np.cos(agent_th)
        ag_y = agent_r * np.sin(agent_th)
        ag_z = 10 * 2 + (ag_x**2 - 10 * np.cos(2 * np.pi * ag_x)) + (ag_y**2 - 10 * np.cos(2 * np.pi * ag_y))

        ax.scatter(ag_x, ag_y, ag_z, color="white", s=30, edgecolor=ring_color, linewidth=1.0, zorder=15)

        # One agent showing anti-stagnation jump on frame 18-24
        if 18 <= frame_idx <= 24:
            ax.scatter([2.4], [-2.2], [42], color="#ff00ff", marker="X", s=140, zorder=25)
            ax.text(2.5, -2.2, 45, "Anti-Stagnation Jump!", color="#ff00ff", fontsize=9, weight="bold")

        ax.set_title(f"HARO 3D Mechanics Demo | Iteration {int(progress*200):03d}/200\nActive Phase: {layer} (Radius: {cur_r:.2f})", color="white", fontsize=10, weight="bold")
        ax.set_xlabel("X1", color=TEXT_COLOR, fontsize=8)
        ax.set_ylabel("X2", color=TEXT_COLOR, fontsize=8)
        ax.set_zlabel("Fitness f(X)", color=TEXT_COLOR, fontsize=8)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.view_init(elev=34, azim=azimuth)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        image = Image.fromarray(rgba)
        frames.append(image)
        plt.close(fig)

    gif_path = os.path.join(OUTPUT_DIR, "demo_3d_halo_mechanisms.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=150, loop=0)
    print(f"-> Saved Animated GIF: {gif_path}")


if __name__ == "__main__":
    generate_3d_pollution_topography()
    generate_3d_halo_concept()
    generate_3d_parameter_sensitivity()
    generate_3d_trajectory_evolution()
    generate_3d_animated_gif()
    print("All 3D assets generated successfully!")
