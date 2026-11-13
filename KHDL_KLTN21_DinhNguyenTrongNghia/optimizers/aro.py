from __future__ import annotations

import time
import numpy as np

from .base import BaseOptimizer, OptimizerConfig, AgentState, OptimizationResult


class ARO(BaseOptimizer):
    """Baseline Artificial Rabbits Optimization adapted to grid station placement.

    Original ARO has three core mechanisms:
      1. detour foraging -> exploration,
      2. random hiding -> exploitation,
      3. energy shrink -> transition between the two.

    The continuous ARO proposal is decoded to the discrete monitoring-grid
    problem through BaseOptimizer._decode_and_repair().
    """

    def __init__(self, problem, config: OptimizerConfig):
        super().__init__(problem, config)

    def _energy(self, t: int) -> float:
        r = max(self.rng.random(), 1e-12)
        return 4.0 * (1.0 - t / self.config.max_iterations) * np.log(1.0 / r)

    def _detour_foraging(self, i, population, t):
        agent = population[i]
        choices = [j for j in range(len(population)) if j != i]
        j = int(self.rng.choice(choices))
        other = population[j].vector

        r2 = self.rng.random()
        r3 = self.rng.random()
        L = (
            np.e - np.exp(((t + 1) / self.config.max_iterations) ** 2)
        ) * np.sin(2.0 * np.pi * r2)

        mask = np.zeros(self.dimension)
        n_active = max(1, int(np.ceil(r3 * self.dimension)))
        active = self.rng.choice(self.dimension, size=n_active, replace=False)
        mask[active] = 1.0

        noise = self.rng.normal(0.0, 1.0, self.dimension)
        R = L * mask

        candidate = (
            other
            + R * (agent.vector - other)
            + np.round(0.5 * (0.05 + self.rng.random())) * noise
        )
        return candidate

    def _random_hiding(self, agent, t):
        r4 = self.rng.random()
        r5 = self.rng.random()

        g = np.zeros(self.dimension)
        g[min(int(r5 * self.dimension), self.dimension - 1)] = 1.0

        H = (
            (self.config.max_iterations - t + 1)
            / self.config.max_iterations
        )

        burrow = agent.vector + H * r4 * g * agent.vector
        R = self.rng.uniform(-1.0, 1.0)
        candidate = agent.vector + R * (r4 * burrow - agent.vector)
        return candidate

    def optimize(self) -> OptimizationResult:
        start = time.perf_counter()

        population = self._initialize()
        best = self._best(population)
        convergence = [best.fitness]
        fe_trace = [(self.problem.fe_count, best.fitness)]

        for t in range(self.config.max_iterations):
            if self._budget_exhausted():
                break

            for i, agent in enumerate(population):
                if self._budget_exhausted():
                    break

                A = self._energy(t)
                if A > 1.0:
                    vector = self._detour_foraging(i, population, t)
                    phase = "exploration"
                else:
                    vector = self._random_hiding(agent, t)
                    phase = "exploitation"

                vector = np.clip(vector, self.lb_vec, self.ub_vec)
                solution, fitness, details = self._evaluate_vector(vector)

                old_fitness = agent.fitness
                improved = self._update_agent(
                    agent, vector, solution, fitness, details
                )
                agent.energy = A
                agent.phase = phase

                if improved:
                    agent.stagnation = 0
                elif fitness >= old_fitness:
                    agent.stagnation += 1

                # Update best and trace by FE
                if fitness < best.fitness:
                    best = agent
                fe_trace.append((self.problem.fe_count, best.fitness))

            best = self._best(population)
            convergence.append(best.fitness)

            if self.config.verbose:
                print(
                    f"[ARO] iter={t+1:04d}/{self.config.max_iterations} "
                    f"FE={self.problem.fe_count} "
                    f"best_fitness={best.fitness:.8f}"
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
            algorithm="ARO",
            seed=self.config.seed,
            fe_convergence=np.asarray(fe_trace),
            metadata={
                "population_size": self.config.population_size,
                "max_iterations": self.config.max_iterations,
                "actual_fe": self.problem.fe_count,
                "max_fes": self.config.max_fes,
            },
        )
