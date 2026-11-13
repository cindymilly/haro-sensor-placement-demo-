from __future__ import annotations

import time
import numpy as np

from .aro import ARO
from .base import OptimizerConfig, OptimizationResult


class HARO(ARO):
    """Halo-enhanced Artificial Rabbits Optimization.

    HARO keeps the ARO baseline intact and adds a multi-layer Halo around the
    current global best solution.

    Proposed HALO mechanism:
      - inner halo: local exploitation around the best;
      - middle halo: balanced search around the best;
      - outer halo: broader exploration around the best.

    The layer radius contracts over time. A stagnating population is allowed
    to activate a wider layer again. This is a research-specific extension,
    not a formula taken from the original ARO paper.
    """

    def __init__(
        self,
        problem,
        config: OptimizerConfig,
        halo_inner=0.05,
        halo_middle=0.15,
        halo_outer=0.35,
        halo_strength=0.50,
        stagnation_threshold=8,
    ):
        super().__init__(problem, config)

        if not (0 < halo_inner < halo_middle < halo_outer):
            raise ValueError(
                "Require 0 < halo_inner < halo_middle < halo_outer."
            )
        if halo_strength < 0:
            raise ValueError("halo_strength must be >= 0.")
        if stagnation_threshold < 1:
            raise ValueError("stagnation_threshold must be >= 1.")

        self.halo_inner = float(halo_inner)
        self.halo_middle = float(halo_middle)
        self.halo_outer = float(halo_outer)
        self.halo_strength = float(halo_strength)
        self.stagnation_threshold = int(stagnation_threshold)

    def _select_layer(self, t, stagnation):
        progress = t / max(1, self.config.max_iterations - 1)

        # Default schedule contracts outer -> middle -> inner.
        if progress < 1 / 3:
            layer = "outer"
        elif progress < 2 / 3:
            layer = "middle"
        else:
            layer = "inner"

        # Stagnation re-opens a wider halo.
        if stagnation >= self.stagnation_threshold:
            layer = "outer"
        elif stagnation >= self.stagnation_threshold // 2:
            layer = "middle"

        return layer

    def _halo_radius(self, layer, t):
        base = {
            "inner": self.halo_inner,
            "middle": self.halo_middle,
            "outer": self.halo_outer,
        }[layer]

        progress = t / max(1, self.config.max_iterations)
        # In continuous optimization, radius must contract to zero for fine exploitation
        if getattr(self.problem, "is_continuous", False):
            contraction = (1.0 - progress) ** 2
        else:
            contraction = 1.0 - 0.5 * progress
        return base * self.domain_diagonal * contraction

    def _halo_update(self, agent, best, layer, t, proposal=None):
        radius = self._halo_radius(layer, t)

        # Attraction toward best.
        attraction = self.rng.uniform(0.0, 1.0)

        # Random direction creates local/global exploration.
        direction = self.rng.normal(0.0, 1.0, self.dimension)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm

        base_vec = proposal if proposal is not None else agent.vector

        # Repulsion term prevents every rabbit from collapsing immediately
        # onto the same decoded grid points or identical local optima.
        repulsion = np.zeros(self.dimension)
        distance = base_vec - best.vector
        dist_norm = np.linalg.norm(distance)
        if dist_norm > 1e-12:
            repulsion = distance / dist_norm

        # Outer layer gets more random movement; inner gets more attraction.
        layer_factor = {
            "inner": 0.85,
            "middle": 0.50,
            "outer": 0.20,
        }[layer]

        candidate = (
            base_vec
            + self.halo_strength
            * attraction
            * (best.vector - base_vec)
            + radius * (
                layer_factor * direction
                + (1.0 - layer_factor) * 0.20 * repulsion
            )
        )

        return candidate

    def optimize(self) -> OptimizationResult:
        start = time.perf_counter()

        population = self._initialize()
        best = self._best(population)
        convergence = [best.fitness]
        fe_trace = [(self.problem.fe_count, best.fitness)]

        halo_history = []

        is_continuous = getattr(self.problem, "is_continuous", False)

        for t in range(self.config.max_iterations):
            if self._budget_exhausted():
                break

            for i, agent in enumerate(population):
                if self._budget_exhausted():
                    break

                A = self._energy(t)

                # Keep the original ARO phase switch.
                if A > 1.0:
                    proposal = self._detour_foraging(i, population, t)
                    phase = "exploration"
                else:
                    proposal = self._random_hiding(agent, t)
                    phase = "exploitation"

                # Apply HALO on top of the ARO proposal.
                layer = self._select_layer(t, agent.stagnation)

                if is_continuous:
                    # In continuous problems: evaluate ARO proposal and Halo refinement, select superior point
                    p_clip = np.clip(proposal, self.lb_vec, self.ub_vec)
                    sol1, fit1, det1 = self._evaluate_vector(p_clip)
                    if self._budget_exhausted():
                        vector, solution, fitness, details = p_clip, sol1, fit1, det1
                    else:
                        halo_cand = self._halo_update(agent, best, layer, t, proposal=p_clip)
                        h_clip = np.clip(halo_cand, self.lb_vec, self.ub_vec)
                        sol2, fit2, det2 = self._evaluate_vector(h_clip)

                        if fit2 < fit1:
                            vector, solution, fitness, details = h_clip, sol2, fit2, det2
                        else:
                            vector, solution, fitness, details = p_clip, sol1, fit1, det1
                else:
                    vector = self._halo_update(
                        agent,
                        best,
                        layer,
                        t,
                        proposal=proposal,
                    )
                    vector = np.clip(
                        vector,
                        self.lb_vec,
                        self.ub_vec,
                    )
                    solution, fitness, details = self._evaluate_vector(vector)

                improved = self._update_agent(
                    agent, vector, solution, fitness, details
                )

                agent.energy = A
                agent.phase = phase
                agent.halo_layer = layer

                if improved:
                    agent.stagnation = 0

                if fitness < best.fitness:
                    best = agent
                fe_trace.append((self.problem.fe_count, best.fitness))

            best = self._best(population)
            convergence.append(best.fitness)

            layer_counts = {
                layer: sum(a.halo_layer == layer for a in population)
                for layer in ("inner", "middle", "outer")
            }
            halo_history.append(layer_counts)

            if self.config.verbose:
                print(
                    f"[HARO] iter={t+1:04d}/{self.config.max_iterations} "
                    f"FE={self.problem.fe_count} "
                    f"best_fitness={best.fitness:.8f} "
                    f"halo={best.halo_layer}"
                )

        runtime = time.perf_counter() - start
        details = self.problem.evaluate_with_details(best.solution, count_fe=False)

        return OptimizationResult(
            best_solution=best.solution,
            best_fitness=best.fitness,
            best_details=details,
            convergence=np.asarray(convergence),
            evaluations=self.problem.fe_count,
            runtime_seconds=runtime,
            algorithm="HARO",
            seed=self.config.seed,
            fe_convergence=np.asarray(fe_trace),
            metadata={
                "population_size": self.config.population_size,
                "max_iterations": self.config.max_iterations,
                "actual_fe": self.problem.fe_count,
                "max_fes": self.config.max_fes,
                "halo_inner": self.halo_inner,
                "halo_middle": self.halo_middle,
                "halo_outer": self.halo_outer,
                "halo_strength": self.halo_strength,
                "stagnation_threshold": self.stagnation_threshold,
                "halo_history": halo_history,
            },
        )
