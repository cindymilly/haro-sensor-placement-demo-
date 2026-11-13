from __future__ import annotations

import time
import numpy as np

from .base import BaseOptimizer, OptimizerConfig, OptimizationResult


class MVO(BaseOptimizer):
    """Multi-Verse Optimizer (MVO) adapted to grid station placement.

    Core Concepts (Mirjalili et al., 2016):
      - Universes: candidate solution vectors representing K station coordinates.
      - Inflation Rate: normalized fitness of each universe (higher for lower fitness in minimization).
      - White/Black Hole Tunnels: exchange coordinates between universes via roulette-wheel selection.
      - Wormhole Existence Probability (WEP): linearly increasing probability of teleporting towards
        the best universe.
      - Travelling Distance Rate (TDR): decreasing factor controlling the perturbation radius
        around the best universe.
    """

    def __init__(
        self,
        problem,
        config: OptimizerConfig,
        wep_min: float = 0.2,
        wep_max: float = 1.0,
    ):
        super().__init__(problem, config)
        self.wep_min = float(wep_min)
        self.wep_max = float(wep_max)

    def _roulette_wheel_selection(self, probabilities: np.ndarray) -> int:
        cumulative = np.cumsum(probabilities)
        p = self.rng.random()
        idx = np.searchsorted(cumulative, p)
        return min(int(idx), len(probabilities) - 1)

    def optimize(self) -> OptimizationResult:
        start = time.perf_counter()

        population = self._initialize()
        best_agent = self._best(population)

        best_vector = best_agent.vector.copy()
        best_solution = best_agent.solution
        best_fitness = best_agent.fitness
        convergence = [best_fitness]
        fe_trace = [(self.problem.fe_count, best_fitness)]

        lb_full = self.lb_vec
        ub_full = self.ub_vec

        for t in range(self.config.max_iterations):
            if self._budget_exhausted():
                break

            # Update WEP and TDR
            progress = t / max(1, self.config.max_iterations - 1)
            wep = self.wep_min + progress * (self.wep_max - self.wep_min)
            tdr = 1.0 - (t / max(1, self.config.max_iterations)) ** (1.0 / 6.0)

            # Compute normalized inflation rates (higher value = better universe = lower fitness)
            fitnesses = np.array([a.fitness for a in population])
            f_max = np.max(fitnesses)
            f_min = np.min(fitnesses)
            denom = f_max - f_min
            if denom > 1e-12:
                norm_inflation = (f_max - fitnesses) / denom + 1e-4
            else:
                norm_inflation = np.ones(len(population))
            probabilities = norm_inflation / np.sum(norm_inflation)

            # Evolve universes
            for i, agent in enumerate(population):
                if self._budget_exhausted():
                    break

                new_vector = agent.vector.copy()

                for j in range(self.dimension):
                    # White/Black hole tunnel exchange
                    r1 = self.rng.random()
                    if r1 < probabilities[i]:
                        selected_idx = self._roulette_wheel_selection(probabilities)
                        new_vector[j] = population[selected_idx].vector[j]

                    # Wormhole tunnels around best universe
                    r2 = self.rng.random()
                    if r2 < wep:
                        r3 = self.rng.random()
                        r4 = self.rng.random()
                        perturbation = tdr * ((ub_full[j] - lb_full[j]) * r4 + lb_full[j])
                        if r3 < 0.5:
                            new_vector[j] = best_vector[j] + perturbation
                        else:
                            new_vector[j] = best_vector[j] - perturbation

                new_vector = np.clip(new_vector, lb_full, ub_full)
                solution, fitness, details = self._evaluate_vector(new_vector)

                self._update_agent(agent, new_vector, solution, fitness, details)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_vector = new_vector.copy()
                    best_solution = solution

                fe_trace.append((self.problem.fe_count, best_fitness))

            convergence.append(best_fitness)

            if self.config.verbose:
                print(
                    f"[MVO] iter={t+1:04d}/{self.config.max_iterations} "
                    f"FE={self.problem.fe_count} "
                    f"best_fitness={best_fitness:.8f} WEP={wep:.4f}"
                )

        runtime = time.perf_counter() - start
        details = self.problem.evaluate_with_details(best_solution, count_fe=False)

        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            best_details=details,
            convergence=np.asarray(convergence),
            evaluations=self.problem.fe_count,
            runtime_seconds=runtime,
            algorithm="MVO",
            seed=self.config.seed,
            fe_convergence=np.asarray(fe_trace),
            metadata={
                "population_size": self.config.population_size,
                "max_iterations": self.config.max_iterations,
                "actual_fe": self.problem.fe_count,
                "max_fes": self.config.max_fes,
                "wep_min": self.wep_min,
                "wep_max": self.wep_max,
            },
        )
