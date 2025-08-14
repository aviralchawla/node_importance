from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Set, Optional
import numpy as np

from node_importance.network import Network

####### BASE CLASS FOR ALL CONTAGION MODELS #######

class DiffusionModel(ABC):
    """
    Abstract base class for all diffusion models.
    To create a new diffusion model, inherit from this class and implement the abstract methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the diffusion model with a network.

        Parameters:
        ----------
        **kwargs : dict
            Additional parameters specific to the diffusion model.
        """
        self.params = kwargs

    @abstractmethod
    def _spread(self, network: Network, recently_infected: Set[int], active_nodes: Set[int], immunized_nodes: Set[int]) -> Set[int]:
        """
        Spread the contagion from recently infected nodes to their neighbors.

        Parameters:
        ----------
        network : Network
            The network on which the diffusion model will be applied.
        recently_infected : Set[int]
            Nodes that were infected in the last time step.
        active_nodes : Set[int]
            Currently active (infected) nodes.
        immunized_nodes : Set[int]
            Immunized nodes that cannot be infected.

        Returns:
        -------
        Set[int]
            Newly infected nodes in this time step.
        """
        pass

    def run(self, network: Network, infected_init: Set[int], immunized_init: Optional[Set[int]] = None) -> Tuple[Set[int], List[Set[int]]]:
        """
        Runs a single simulation of the diffusion model on the given network to completion.

        Parameters:
        ----------
        network : Network
            The network on which the diffusion model will be applied.
        infected_init : Set[int]
            Initial infected nodes.
        immunized_init : Set[int], optional
            Initial immunized nodes. Defaults to None.

        Returns:
        -------
        Tuple[Set[int], List[Set[int]]]
            Final infected nodes and a list of states at each time step.
        """
        
        active_nodes = infected_init.copy()
        immunized_nodes = immunized_init.copy() if immunized_init else set()

        assert active_nodes.isdisjoint(immunized_nodes), "Active nodes and immunized nodes must be disjoint."
        assert len(active_nodes) > 0, "At least one node must be initially infected."
        assert len(active_nodes) <= network.number_of_nodes(), "Initial infected nodes cannot exceed the number of nodes in the network."

        history = [active_nodes.copy()]
        
        recently_infected = active_nodes.copy()

        while recently_infected:
            newly_infected = self._spread(network, recently_infected, active_nodes, immunized_nodes)
            if not newly_infected:
                break
            
            active_nodes.update(newly_infected)
            history.append(active_nodes.copy())
            recently_infected = newly_infected.copy()

        return active_nodes, history
    
    def __call__(self, network: Network, infected_init: Set[int], immunized_init: Optional[Set[int]] = None) -> Tuple[Set[int], List[Set[int]]]:
        return self.run(network, infected_init, immunized_init)

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


######## INDEPENDENT CASCADE #######

class IndependentCascade(DiffusionModel):
    """
    The Independent Cascade (IC) model is a commonly used model for diffusion in a network. 
    In this model, the transmission of information or disease from one node to another occurs with a fixed probability, β. 
    This means that if a node is infectious, the probability of infecting a connected node is solely dependent on this fixed probability, and not on any other metric.
    """

    def __init__(self, p: float, **kwargs):
        """
        Initialize the Independent Cascade model.

        Parameters:
        ----------
        p : float
            Probability of infection for each edge.
        **kwargs : dict
            Additional parameters specific to the diffusion model.
        """
        super().__init__(**kwargs)
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")
        self.p = p
        self.params['infection_probability'] = p

    def _spread(self, network: Network, recently_infected: Set[int], active_nodes: Set[int], immunized_nodes: Set[int]) -> Set[int]:
        """
        Spread the infection from recently infected nodes to their neighbors.
        Contagion occurs with a fixed probability p for each edge.
        """
        uninfectable = active_nodes.union(immunized_nodes)

        susceptible = [
            v for u in recently_infected
            for v in network.get_neighbors(u)
            if v not in uninfectable
            ]

        if not susceptible:
            return set()

        probabilities = [self.p] * len(susceptible)
        infection_outcomes = np.random.rand(len(probabilities)) < probabilities

        new_infected = {susceptible[i] for i in range(len(susceptible)) if infection_outcomes[i]}

        return new_infected


######## WEIGHTED CASCADE #######

class WeightedCascade(DiffusionModel):
    """
    The Weighted Cascade (WC) model is another method we employ to measure the spread in the network. 
    Similar to the IC model, we use a stochastic process to simulate the spread. 
    However, the probability of infection is depends on the weight (w) of an edge between two nodes.
    In contrast to the IC model, where the infection probability is fixed. 
    """
    
    def __init__(self, network: Network, weights: Dict[Tuple[int, int], float] = None, **kwargs):
        """
        Initialize the Weighted Cascade model.

        Parameters:
        ----------
        network : Network
            The network on which the diffusion model will be applied.
        weights : Dict[Tuple[int, int], float]
            A dictionary mapping edges (u, v) to their infection probabilities.
        **kwargs : dict
            Additional parameters specific to the diffusion model.
        """
        super().__init__(**kwargs)
        self.network = network
        if weights is None:
            if network.directed:
                weights = self._init_weights_in_degree()
            else:
                weights = self._init_random_weights()

        self._validate_weights(weights)
        self.weights = weights

        self.params['weights'] = weights

    def _validate_weights(self, weights: Dict[Tuple[int, int], float]):
        """
        All weights must be in the range [0, 1].
        """
        for (u, v), weight in weights.items():
            if not (0 <= weight <= 1):
                raise ValueError(f"Invalid weight {weight} for edge ({u}, {v}). Weights must be in the range [0, 1].")

    def _init_weights_in_degree(self) -> Dict[Tuple[int, int], float]:
        """
        Initialize weights based on the in-degree of nodes in the network.
        Weight of edge (u, v) is 1 / in-degree(v).

        Returns:
        -------
        Dict[Tuple[int, int], float]
            A dictionary mapping edges (u, v) to their infection probabilities based on in-degree.
        """
        weights = {}
        for u, v in self.network.edges():
            in_degree = self.network.get_in_degree(v)
            if in_degree > 0:
                weights[(u, v)] = 1 / in_degree
        return weights

    def _init_random_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Initialize random weights for edges in the network.

        Returns:
        -------
        Dict[Tuple[int, int], float]
            A dictionary mapping edges (u, v) to random infection probabilities.
        """
        weights = {}
        for u, v in self.network.edges():
            weights[(u, v)] = np.random.uniform(0, 1)
        return weights

    def _spread(self, network: Network, recently_infected: Set[int], active_nodes: Set[int], immunized_nodes: Set[int]) -> Set[int]:
        """
        Spread the contagion from recently infected nodes to their neighbors based on edge weights.
        """
        uninfectable = active_nodes.union(immunized_nodes)

        susceptible = [
            v for u in recently_infected
            for v in network.get_neighbors(u)
            if v not in uninfectable and (u, v) in self.weights
        ]

        if not susceptible:
            return set()

        probabilities = [
                self.weights[(u, v)] for u in recently_infected
                for v in network.get_neighbors(u) if v in susceptible
                ]

        infection_outcomes = np.random.rand(len(probabilities)) < probabilities
        new_infected = {susceptible[i] for i in range(len(susceptible)) if infection_outcomes[i]}

        return new_infected
