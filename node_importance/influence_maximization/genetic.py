import logging
from typing import Set
from tqdm import tqdm
import os
import random
import numpy as np

from node_importance.diffusion import DiffusionModel
from node_importance.network import Network

logger = logging.getLogger(__name__)

class Genetic:
    """
    The Genetic algorithm for influence maximzation. The algorithm starts with a randomly initialzied population of individuals (nodes).
    We then iteratively select individuals based on their fitness (influence), perform crossover and mutation to create new individuals,
    and replace the old population with the new one. This process continues until we reach a specified number of generations.

    The algorithm was proposed by ...
    """

    def __init__(self, diffusion_model: DiffusionModel, num_samples: int = 10,
                 population_size: int = 100, mutation_rate: float = 0.1,
                 tournament_type: str = 'classic', mutation_type: str = 'random',
                 crossover_type: str = 'one-point', elites: int = 5,
                 tournament_size: int = 5):

        self.diffusion_model = diffusion_model
        self.num_samples = num_samples
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_type = tournament_type
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.elites = elites
        self.tournament_size = tournament_size

    class Individual:
        """
        Each individual in our population
        """
        def __init__(self, genome: Set[int], fitness: float = -1.0):
            self.genome = genome
            self.fitness = fitness

    def _initialize_population(self, network: Network, k: int) -> list[Individual]:
        """
        Initialize the population with random individuals
        """
        population = []
        for _ in range(self.population_size):
            genome = set(np.random.choice(network.nodes(), size=k, replace=False))
            individual = self.Individual(genome)
            population.append(individual)
        return population

    def _calculate_individual_fitness(args):
        """
        Static method to calculate fitness for a single individual.
        This allows for easy parallelization.
        """
        network, diffusion_model, genome, num_samples = args
        
        total_influence = 0.0
        for rep in range(num_samples):
            infected = genome.copy()
            spread, history = diffusion_model(network, infected)
            total_influence += len(spread)

        average_influence = total_influence / num_samples
        return average_influence

    def _estimate_influence(self, network: Network, genome: Set[int]) -> float:
        """
        Estimate the influence of a given genome using the contagion model multiple times.

        Parameters:
        ----------
        network : Network
            The network on which the influence maximization is performed.
        genome : Set[int]
            The set of nodes that are seeded for the contagion process.
        Returns:
        -------
        float
            The average influence of the genome over multiple contagion runs.
        """
        args = (network, self.diffusion_model, genome, self.num_samples)
        return self._calculate_individual_fitness(args)

    ##### TOURNAMENTS #####

    def _classic_tournament_selection(self, population: list[Individual]) -> Individual:
        """
        Sample n individuals from the population and select the best
        """
        selected = random.sample(population, self.tournament_size)
        return sorted(selected, key=lambda ind: ind.fitness, reverse=True)[0]

    def _tournament_selection(self, population: list[Individual]) -> Individual:
        if self.tournament_type == 'classic':
            return self._classic_tournament_selection(population)
        else:
            raise ValueError(f"Unknown tournament type: {self.tournament_type}. "
                             f"Supported types: 'classic'.")

    ##### CROSSOVER #####

    def _one_point_crossover(self, parent_1: Individual, parent_2: Individual) -> tuple[Individual, Individual]:
        """
        Perform one-point crossover between two parents to create two children.
        """
        k = len(parent_1.genome)
        crossover_point = random.randint(1, k - 1)

        p1_1, p1_2 = list(parent_1.genome)[:crossover_point], list(parent_1.genome)[crossover_point:]
        p2_1, p2_2 = list(parent_2.genome)[:crossover_point], list(parent_2.genome)[crossover_point:]

        child_1_genome = set(p1_1 + p2_2)
        child_2_genome = set(p2_1 + p1_2)

        return self.Individual(child_1_genome), self.Individual(child_2_genome)

    def _crossover(self, parent_1: Individual, parent_2: Individual) -> tuple[Individual, Individual]:
        if self.crossover_type == 'one-point':
            return self._one_point_crossover(parent_1, parent_2)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}. "
                             f"Supported types: 'one-point'.")

    ##### MUTATION #####

    def _random_mutation(self, network: Network, individual: Individual) -> Individual:
        """
        Randomly mutate one node in the individual's genome.
        """
        rand_indx = random.randint(0, len(individual.genome) - 1)
        new_node = random.choice(list(set(network.nodes()) - individual.genome))
        genome_list = list(individual.genome)
        genome_list[rand_indx] = new_node
        return self.Individual(set(genome_list))
    
    def _neighbor_hop_mutation(self, network: Network, individual: Individual) -> Individual:
        return individual


    def _mutate(self, network: Network, individual: Individual) -> Individual:
        if self.mutation_type == 'random':
            return self._random_mutation(network, individual)
        elif self.mutation_type == 'neighbor_hop':
            return self._neighbor_hop_mutation(network, individual)
        else:
            raise ValueError(f"Unknown mutation type: {self.mutation_type}. "
                             f"Supported types: 'random', 'neighbor_hop'.")


    def _get_next_generation(self, network: Network, population: list[Individual]) -> list[Individual]:
        """
        Generates the next generation of individuals based on current population and
        Genetic Algorithm principles such as selection, crossover, and mutation.

        Parameters:
        ----------
        population : list[Individual]
            The current population of individuals.
        Returns:
        -------
        list[Individual]
            The next generation of individuals.
        """
        next_generation = []

        if self.elites > 0:
            elites = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:self.elites]
            next_generation.extend(elites)

        while len(next_generation) < self.population_size:
            parent_1, parent_2 = self._tournament_selection(population), \
                    self._tournament_selection(population)
            child_1, child_2 = self._crossover(parent_1, parent_2)

            child_1, child_2 = self._mutate(network, child_1), self._mutate(network, child_2)

            next_generation.append(child_1)
            next_generation.append(child_2)

        return next_generation[:self.population_size]

    def fit(self, network: Network, k: int, num_generations: int = 100) -> Set[int]:
        """
        Find top-k nodes using genetic algorithm and provided mutation, tournament, and crossover strategies.

        Parameters:
        ----------
        network : Network
            The network on which the influence maximization is performed.
        k : int
            The number of nodes to select.
        num_generations : int, optional
            The number of generations to run the genetic algorithm (default is 100).
        Returns:
        -------
        Set[int]
            The set of selected nodes.
        """

        logger.info("Starting Genetic Algorithm with the following parameters:")
        logger.info(f"Population Size: {self.population_size}, "
                    f"Mutation Rate: {self.mutation_rate}, "
                    f"Tournament Type: {self.tournament_type}, "
                    f"Mutation Type: {self.mutation_type}, "
                    f"Crossover Type: {self.crossover_type}, "
                    f"Elites: {self.elites}, "
                    f"Tournament Size: {self.tournament_size}, "
                    f"Number of Generations: {num_generations}")

        population = self._initialize_population(network, k)
        best_individual = self.Individual(set(), -1.0)

        for gen in tqdm(range(num_generations), desc="Generations"):
            if gen % 10 == 0:
                logger.info(f"Generation {gen}/{num_generations}")
                logger.info(f"Best Fitness: {best_individual.fitness if best_individual else 'N/A'}")

            # Find individuals that need fitness calculation
            individuals_to_evaluate = [ind for ind in population if ind.fitness == -1.0]
            
            if individuals_to_evaluate:
                for individual in individuals_to_evaluate:
                    individual.fitness = self._estimate_influence(network, individual.genome)

            population.sort(key=lambda ind: ind.fitness, reverse=True)

            if best_individual is None or population[0].fitness > best_individual.fitness:
                best_individual = population[0]

            population = self._get_next_generation(network, population)

        logger.info(f"Best Individual Genome: {best_individual.genome}, Fitness: {best_individual.fitness}")
        return best_individual.genome



