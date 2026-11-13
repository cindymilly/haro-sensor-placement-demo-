from __future__ import annotations

import time
import numpy as np

from .base import BaseOptimizer, OptimizerConfig, OptimizationResult


class PSO(BaseOptimizer):
    """Particle Swarm Optimization adapted to grid station placement.

    Formulation:
      - Each particle represents coordinates of K stations in continuous space (2K dimensions).
      - Velocity update combines inertia, personal best memory, and global best swarm guidance:
          v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i(t)) + c2 * r2 * (gbest - x_i(t))
      - Inertia weight w linearly decreases over iterations from w_max to w_min to balance
        global exploration with fine local convergence.
      - Discrete candidate mapping and geometric constraints are handled via BaseOptimizer's
        decoding and repair pipeline.
    """

    def __init__(
        self,
        problem,
        config: OptimizerConfig,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max_ratio: float = 0.2,
    ):
        super().__init__(problem, config)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.v_max_ratio = float(v_max_ratio)

    def optimize(self) -> OptimizationResult:
        start = time.perf_counter()

        population = self._initialize()
        best_agent = self._best(population)

        # Personal best positions and fitnesses
        pbests = [agent.vector.copy() for agent in population]
        pbest_fitnesses = [agent.fitness for agent in population]

        # Global best
        gbest_vector = best_agent.vector.copy()
        gbest_solution = best_agent.solution
        gbest_fitness = best_agent.fitness
        convergence = [gbest_fitness]
        fe_trace = [(self.problem.fe_count, gbest_fitness)]

        lb_full = self.lb_vec
        ub_full = self.ub_vec
        v_max = self.v_max_ratio * (ub_full - lb_full)

        # Initialize velocities randomly within [-v_max, v_max]
        velocities = [
            self.rng.uniform(-v_max, v_max, size=self.dimension)
            for _ in range(self.config.population_size)
        ]

        for t in range(self.config.max_iterations):
            if self._budget_exhausted():
                break

            # Dynamic inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (t / max(1, self.config.max_iterations - 1))

            for i, agent in enumerate(population):
                if self._budget_exhausted():
                    break

                r1 = self.rng.random(self.dimension)
                r2 = self.rng.random(self.dimension)

                # Velocity update
                new_v = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbests[i] - agent.vector)
                    + self.c2 * r2 * (gbest_vector - agent.vector)
                )
                new_v = np.clip(new_v, -v_max, v_max)
                velocities[i] = new_v

                # Position update
                new_pos = agent.vector + new_v
                new_pos = np.clip(new_pos, lb_full, ub_full)

                # Evaluation and discrete repair
                solution, fitness, details = self._evaluate_vector(new_pos)
                self._update_agent(agent, new_pos, solution, fitness, details)

                # Personal best update
                if fitness < pbest_fitnesses[i]:
                    pbests[i] = new_pos.copy()
                    pbest_fitnesses[i] = fitness

                # Global best update
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_vector = new_pos.copy()
                    gbest_solution = solution

                fe_trace.append((self.problem.fe_count, gbest_fitness))

            convergence.append(gbest_fitness)

            if self.config.verbose:
                print(
                    f"[PSO] iter={t+1:04d}/{self.config.max_iterations} "
                    f"FE={self.problem.fe_count} "
                    f"best_fitness={gbest_fitness:.8f}"
                )

        runtime = time.perf_counter() - start
        details = self.problem.evaluate_with_details(gbest_solution, count_fe=False)

        return OptimizationResult(
            best_solution=gbest_solution,
            best_fitness=gbest_fitness,
            best_details=details,
            convergence=np.asarray(convergence),
            evaluations=self.problem.fe_count,
            runtime_seconds=runtime,
            algorithm="PSO",
            seed=self.config.seed,
            fe_convergence=np.asarray(fe_trace),
            metadata={
                "population_size": self.config.population_size,
                "max_iterations": self.config.max_iterations,
                "actual_fe": self.problem.fe_count,
                "max_fes": self.config.max_fes,
                "w_max": self.w_max,
                "w_min": self.w_min,
                "c1": self.c1,
                "c2": self.c2,
            },
        )
