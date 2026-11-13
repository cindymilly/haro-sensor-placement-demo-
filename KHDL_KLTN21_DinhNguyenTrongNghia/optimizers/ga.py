from __future__ import annotations

import time
import numpy as np

from .base import BaseOptimizer, OptimizerConfig, OptimizationResult


class GA(BaseOptimizer):
    """Real-Coded Genetic Algorithm (GA) adapted to grid station placement.

    Mechanisms:
      - Chromosome: continuous vector representing coordinates of K stations (2K dimensions).
      - Selection: Tournament selection to choose fit parent chromosomes.
      - Crossover: Blend Crossover (BLX-alpha) generating diverse offspring while preserving parent traits.
      - Mutation: Adaptive Gaussian mutation introducing local perturbations to prevent stagnation.
      - Elitism: Strictly preserves the top elite individuals into the next generation.
      - Discrete candidate mapping and constraint repair handled by BaseOptimizer.
    """

    def __init__(
        self,
        problem,
        config: OptimizerConfig,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        alpha: float = 0.3,
        tournament_size: int = 3,
        elite_count: int = 2,
    ):
        super().__init__(problem, config)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.alpha = float(alpha)
        self.tournament_size = max(2, int(tournament_size))
        self.elite_count = max(1, min(elite_count, config.population_size // 2))

    def _tournament_selection(self, population: list) -> int:
        indices = self.rng.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = indices[0]
        best_fit = population[best_idx].fitness
        for idx in indices[1:]:
            if population[idx].fitness < best_fit:
                best_fit = population[idx].fitness
                best_idx = idx
        return int(best_idx)

    def optimize(self) -> OptimizationResult:
        start = time.perf_counter()

        population = self._initialize()
        best_agent = self._best(population)

        best_solution = best_agent.solution
        best_fitness = best_agent.fitness
        convergence = [best_fitness]
        fe_trace = [(self.problem.fe_count, best_fitness)]

        lb_full = self.lb_vec
        ub_full = self.ub_vec
        scale = ub_full - lb_full

        for t in range(self.config.max_iterations):
            if self._budget_exhausted():
                break

            # Sort current population by fitness to identify elites
            population.sort(key=lambda a: a.fitness)
            elites = [population[e] for e in range(self.elite_count)]

            # Adaptive mutation scale: shrinks slightly as generations progress
            mut_scale = 0.10 * scale * (1.0 - 0.5 * (t / max(1, self.config.max_iterations)))

            # Produce exactly population_size offspring to guarantee equal evaluation budget
            new_population = []
            from .base import AgentState

            for _ in range(self.config.population_size):
                if self._budget_exhausted():
                    break

                p1_idx = self._tournament_selection(population)
                p2_idx = self._tournament_selection(population)
                parent1 = population[p1_idx].vector
                parent2 = population[p2_idx].vector

                # Crossover (BLX-alpha)
                if self.rng.random() < self.crossover_rate:
                    min_val = np.minimum(parent1, parent2)
                    max_val = np.maximum(parent1, parent2)
                    diff = max_val - min_val
                    low = min_val - self.alpha * diff
                    high = max_val + self.alpha * diff
                    child = self.rng.uniform(low, high)
                else:
                    child = parent1.copy()

                # Mutation with standard rate (1 / dimension)
                eff_mut_rate = self.mutation_rate if self.mutation_rate > 0 else (1.0 / self.dimension)
                mutate_mask = self.rng.random(self.dimension) < eff_mut_rate
                if np.any(mutate_mask):
                    noise = self.rng.normal(0.0, 1.0, self.dimension) * mut_scale
                    child = child + mutate_mask * noise

                child = np.clip(child, lb_full, ub_full)
                solution, fitness, details = self._evaluate_vector(child)

                new_population.append(
                    AgentState(
                        vector=child,
                        solution=solution,
                        fitness=float(fitness),
                        feasible=details["feasible"],
                    )
                )

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution

                fe_trace.append((self.problem.fe_count, best_fitness))

            if not new_population:
                break

            # Elitism: replace worst offspring with elites if elites are better
            new_population.sort(key=lambda a: a.fitness)
            for e in range(min(self.elite_count, len(new_population))):
                if elites[e].fitness < new_population[-(e + 1)].fitness:
                    new_population[-(e + 1)] = elites[e]

            population = new_population
            current_best = min(population, key=lambda a: a.fitness)

            if current_best.fitness < best_fitness:
                best_fitness = current_best.fitness
                best_solution = current_best.solution

            convergence.append(best_fitness)

            if self.config.verbose:
                print(
                    f"[GA] iter={t+1:04d}/{self.config.max_iterations} "
                    f"FE={self.problem.fe_count} "
                    f"best_fitness={best_fitness:.8f}"
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
            algorithm="GA",
            seed=self.config.seed,
            fe_convergence=np.asarray(fe_trace),
            metadata={
                "population_size": self.config.population_size,
                "max_iterations": self.config.max_iterations,
                "actual_fe": self.problem.fe_count,
                "max_fes": self.config.max_fes,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "alpha": self.alpha,
                "tournament_size": self.tournament_size,
                "elite_count": self.elite_count,
            },
        )
