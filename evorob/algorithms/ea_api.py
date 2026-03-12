import numpy as np
import cma

from evorob.algorithms.base_ea import EA


class EvoAlgAPI(EA):
    """Evolutionary algorithm API wrapper.

    This class provides an interface to wrap any EA framework that uses
    the ask-tell pattern (CMA-ES, pyribs, evosax, etc.).

    Example frameworks to use:
    - CMA-ES: https://github.com/CMA-ES/pycma
    - pyribs: https://github.com/icaros-usc/pyribs/
    - evosax: https://github.com/RobertTLange/evosax/
    - EvoJAX: https://github.com/google/evojax
    """

    def __init__(self, n_params: int, population_size: int = 100, num_generations: int = 100,
                 output_dir: str = "./results/EA", **kwargs):
        """Initialize the evolutionary algorithm.

        Args:
            n_params: Dimensionality of the search space
            population_size: Number of solutions per generation
            num_generations: Number of generations
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments for the EA framework
        """
        # TODO: Initialize your chosen EA framework here
        self.n_params = n_params
        self.n_gen = num_generations
        self.population_size = population_size
        
        # % bookkeeping for base EA
        self.directory_name = output_dir
        self.current_gen = 0
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

        if "sigma" in kwargs:
            sigma = kwargs["sigma"]
        else:
            sigma = 1.0

        opts = {
            "popsize": population_size,
            "maxiter": num_generations,
            "verbose": -1,
            "CMA_mu": population_size // 5,
        }

        x0 = np.zeros(n_params)
        self.cma_es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    def ask(self) -> np.ndarray:
        """Sample population from the algorithm.

        Returns:
            population: Array of shape (population_size, n_params)
                       Each row is a candidate solution
        """
        # TODO: Get new population from your EA
        # Make sure the returned array has shape (population_size, n_params)

        X = np.array(self.cma_es.ask())
        assert X.shape == (self.population_size, self.n_params)
        return X

    def tell(self, population: np.ndarray, fitnesses: np.ndarray, save_checkpoint: bool = False) -> None:
        """Update the algorithm with evaluated population.

        Args:
            population: Array of shape (population_size, n_params)
            fitnesses: Array of shape (population_size,) with fitness values
                      Higher is better (maximization)
            save_checkpoint: Whether to save checkpoint after update
        """
        # TODO: Update your EA with the evaluated population
        # Note: Some algorithms minimize, others maximize.
        # Adjust accordingly (negate fitnesses if needed).
        
        # CMA-ES minimizes objective values, while our framework maximizes fitness.
        # Convert invalid values to very poor fitness, then negate for minimization.
        fitnesses = np.asarray(fitnesses, dtype=float)
        safe_fitnesses = np.where(np.isfinite(fitnesses), fitnesses, -np.inf)
        cma_objective = -safe_fitnesses

        # Replace +inf objectives (from -(-inf)) with a large finite penalty.
        if not np.all(np.isfinite(cma_objective)):
            finite_objectives = cma_objective[np.isfinite(cma_objective)]
            fallback = (np.max(finite_objectives) + 1.0) if finite_objectives.size > 0 else 1e9
            cma_objective = np.where(np.isfinite(cma_objective), cma_objective, fallback)

        # Update CMA-ES with minimization objective
        self.cma_es.tell(population, cma_objective)

        # After updating the EA, do bookkeeping for checkpointing:
        self.full_f.append(fitnesses)
        self.full_x.append(population)
        self.f = fitnesses
        self.x = population
        
        # Track best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.f_best_so_far:
            self.f_best_so_far = fitnesses[best_idx]
            self.x_best_so_far = population[best_idx].copy()
        
        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

