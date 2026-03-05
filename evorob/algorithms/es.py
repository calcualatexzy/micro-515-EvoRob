from typing import Dict

import numpy as np
from evorob.algorithms.base_ea import EA

ES_opts = {
    "min": -4,
    "max": 4,
    "num_parents": 16,
    "num_generations": 100,
    "mutation_sigma": 0.3,
    "min_sigma": 0.1,
    "sigma_decay_rate": 0.95
}


class ES(EA):

    def __init__(self, n_pop, n_params, opts: Dict = ES_opts, log_every: int = 5, output_dir: str = "./results/ES"):
        """
        Evolutionary Strategy

        :param n_pop: population size
        :param n_params: number of parameters
        :param opts: algorithm options
        :log_every: log every n generations
        :param output_dir: output directory Default = "./results/ES"
        """
        # % EA options
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.n_parents = opts["num_parents"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.current_mean = np.array([(self.min + self.max) / 2] * self.n_params)
        self.current_sigma = opts["mutation_sigma"]
        self.min_sigma = opts["min_sigma"]
        self.sigma_decay_rate = opts["sigma_decay_rate"]

        # % bookkeeping
        self.log_every = log_every
        self.directory_name = output_dir
        self.full_x = []
        self.full_f = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

    def ask(self):
        if self.current_gen==0:
            new_population = self.initialise_x0()
        else:
            new_population = self.generate_mutated_offspring(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=False):
        parents_population, parents_fitness = self.sort_and_select_parents(
            solutions, function_values, self.n_parents
        )
        self.update_population_mean(parents_population, parents_fitness, rank=True)
        self.update_sigma()

        #% Some bookkeeping
        self.full_f.append(function_values)
        self.full_x.append(solutions)
        self.x = parents_population
        self.f = parents_fitness

        safe_fitness = np.where(np.isfinite(function_values), function_values, -np.inf)
        finite_population_fitness = safe_fitness[np.isfinite(safe_fitness)]
        if finite_population_fitness.size > 0:
            best_index = np.argmax(safe_fitness)
            if safe_fitness[best_index] > self.f_best_so_far:
                self.f_best_so_far = safe_fitness[best_index]
                self.x_best_so_far = solutions[best_index]
            mean_f = float(np.mean(finite_population_fitness))
            std_f = float(np.std(finite_population_fitness))
            best_f = float(safe_fitness[best_index])
        else:
            best_index = 0
            mean_f = float("nan")
            std_f = float("nan")
            best_f = float("nan")

        if self.current_gen % self.log_every == 0:
            print(f"Best in generation {self.current_gen: 3d}: {best_f:.2f}\n"
                  f"Best fitness so far   : {self.f_best_so_far:.2f}\n"
                  f"Mean pop fitness      : {mean_f:.2f} +- {std_f:.2f}\n"
                  f"Sigma: {self.current_sigma:.2f} \n"
            )
        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def initialise_x0(self):
        """Initialises the first population."""
        # TODO: generate the initial population mean vector (current_mean)
        perturbation = np.random.randn(self.n_pop, self.n_params) * self.current_sigma
        return self.current_mean + perturbation

    def update_sigma(self):
        """Update the perturbation strength (sigma)."""
        # TODO: implement a decay of the sigma value over generations, ensuring it does not go below min_sigma
        self.current_sigma = max(self.current_sigma * self.sigma_decay_rate, self.min_sigma)

    def sort_and_select_parents(self, population, fitness, num_parents):
        """Sorts the population based on fitness and selects the top individuals as parents."""
        # TODO: sort the population and fitness based on fitness values, and select the top num_parents individuals as parents
        num_parents = min(num_parents, len(population))
        safe_fitness = np.where(np.isfinite(fitness), fitness, -np.inf)
        sorted_idx = np.argsort(safe_fitness)[::-1]
        parent_idx = sorted_idx[:num_parents]
        parent_population = population[parent_idx]
        parent_fitness = safe_fitness[parent_idx]

        return parent_population, parent_fitness

    def update_population_mean(self, parent_population, parent_fitness, rank: bool = True):
        # TODO: compute the new population mean as a weighted average of the parent population, where the weights are based on the parent fitness
        # (you can use rank or raw fitness values)
        # Normalise parent fitness scores
        finite_mask = np.isfinite(parent_fitness)
        if not np.any(finite_mask):
            # Keep previous mean when there is no valid fitness signal.
            return self.current_mean

        valid_population = parent_population[finite_mask]

        if rank:
            # Rank-based recombination is more stable than raw-fitness weighting
            # on noisy objectives.
            rank_positions = np.arange(1, valid_population.shape[0] + 1, dtype=float)  # 1 = best
            descending_weights = (valid_population.shape[0] + 1) - rank_positions
            normed_parents_fitness = descending_weights / np.sum(descending_weights)
        else:
            valid_fitness = parent_fitness[finite_mask]
            fitness_shifted = valid_fitness - np.min(valid_fitness)
            if np.sum(fitness_shifted) == 0:
                normed_parents_fitness = np.ones_like(valid_fitness) / len(valid_fitness)
            else:
                normed_parents_fitness = fitness_shifted / np.sum(fitness_shifted)

        # Compute population weighted to the normed fitness scores
        weighted_parents_population = valid_population * normed_parents_fitness[:, np.newaxis]

        # Calculate the sum of weighted parents population
        updated_mean_vector = np.sum(weighted_parents_population, axis=0)
        self.current_mean = updated_mean_vector

        return updated_mean_vector

    def generate_mutated_offspring(self, population_size):
        """Generates a new population by adding Gaussian noise to the current mean."""
        # TODO: generate a new population by adding Gaussian noise to the current mean, where the noise is scaled by the current sigma value
        perturbation = np.random.randn(population_size, self.n_params) * self.current_sigma
        mutated_population = self.current_mean + perturbation

        return mutated_population