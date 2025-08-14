import logging
import heapq
from typing import Set, List, Tuple
from tqdm import tqdm

from node_importance.diffusion import DiffusionModel
from node_importance.network import Network

logger = logging.getLogger(__name__)


class CELF:
    """
    The Cost-Effective Lazy Forward (CELF) algorithm for influence maximization.
    
    CELF optimizes the greedy algorithm by using the submodularity property of influence functions.
    It maintains a priority queue of nodes sorted by their marginal gain and uses lazy evaluation
    to avoid redundant influence calculations.
    
    The algorithm was proposed by Leskovec et al. in "Cost-effective outbreak detection in networks" (2007).
    """

    def __init__(self, diffusion_model: DiffusionModel, num_samples: int = 10):
        """
        Initialize the CELF algorithm for influence maximization.

        Parameters:
        ----------
        diffusion_model : DiffusionModel
            The contagion model to be used for influence maximization.
        num_samples : int
            Number of samples to use for estimating the influence of nodes. Default is 10.
        """
        self.diffusion_model = diffusion_model
        self.num_samples = num_samples

    def _run_single(self, network: Network, infected: Set[int]) -> int:
        """Run one Monte-Carlo simulation and return outbreak size."""
        spread, _ = self.diffusion_model(network, infected)
        return len(spread)

    def _estimate_influence(self, network: Network, node: int, selected_nodes: Set[int]) -> float:
        """
        Estimate the influence of a node by running the contagion model multiple times.

        Parameters:
        ----------
        network : Network
            The network on which the contagion model is applied.
        node : int
            The node for which the influence is estimated.
        selected_nodes : Set[int]
            The set of nodes already selected for influence maximization.

        Returns:
        -------
        float
            The estimated influence of the node.
        """
        results = []
        for _ in range(self.num_samples):
            infected = {node} | selected_nodes
            result = self._run_single(network, infected)
            results.append(result)

        influence = sum(results) / self.num_samples
        return influence

    def _calculate_marginal_gain(self, network: Network, node: int, selected_nodes: Set[int]) -> float:
        """
        Calculate the marginal gain of adding a node to the selected set.

        Parameters:
        ----------
        network : Network
            The network on which the influence maximization is performed.
        node : int
            The node for which marginal gain is calculated.
        selected_nodes : Set[int]
            The set of nodes already selected.

        Returns:
        -------
        float
            The marginal gain of adding the node.
        """
        if not selected_nodes:
            # If no nodes are selected, marginal gain is just the influence of the node
            return self._estimate_influence(network, node, set())
        
        # Calculate influence with and without the node
        influence_with_node = self._estimate_influence(network, node, selected_nodes)
        influence_without_node = self._estimate_influence(network, -1, selected_nodes)  # Use dummy node
        
        return influence_with_node - influence_without_node

    def fit(self, network: Network, k: int) -> Set[int]:
        """
        Find top-k nodes for influence maximization using the CELF algorithm.

        Parameters:
        ----------
        network : Network
            The network on which the influence maximization is performed.
        k : int
            The number of nodes to select for maximizing influence.

        Returns:
        -------
        Set[int]
            A set of node IDs representing the selected nodes for influence maximization.
        """
        logger.info(f"Starting CELF algorithm with k={k}, num_samples={self.num_samples}")
        
        selected_nodes = set()
        remaining_nodes = set(network.nodes())
        
        # Priority queue: (-marginal_gain, node_id, iteration_last_updated)
        # We use negative marginal gain because heapq is a min-heap
        priority_queue: List[Tuple[float, int, int]] = []
        
        # Initialize priority queue with all nodes
        logger.info("Initializing priority queue...")
        for node in tqdm(remaining_nodes, desc="Calculating initial marginal gains"):
            marginal_gain = self._calculate_marginal_gain(network, node, selected_nodes)
            heapq.heappush(priority_queue, (-marginal_gain, node, 0))
        
        total_influence = 0.0
        
        for i in range(k):
            logger.info(f"Selecting node {i + 1}/{k}...")
            
            best_node = None
            best_gain = -float('inf')
            
            # Find the best node using lazy evaluation
            while priority_queue:
                neg_gain, node, last_updated = heapq.heappop(priority_queue)
                marginal_gain = -neg_gain
                
                # If this node was updated in the current iteration, it's the best
                if last_updated == i:
                    best_node = node
                    best_gain = marginal_gain
                    break
                
                # Otherwise, recalculate its marginal gain (lazy evaluation)
                if node in remaining_nodes:
                    new_marginal_gain = self._calculate_marginal_gain(network, node, selected_nodes)
                    heapq.heappush(priority_queue, (-new_marginal_gain, node, i))
            
            if best_node is not None:
                logger.info(f"Selected node {best_node} with marginal gain {best_gain:.2f}")
                selected_nodes.add(best_node)
                remaining_nodes.remove(best_node)
                total_influence += best_gain
                
                # Remove the selected node from any remaining entries in the queue
                priority_queue = [(gain, node, iter_) for gain, node, iter_ in priority_queue 
                                if node != best_node]
                heapq.heapify(priority_queue)
            else:
                logger.warning("Could not find a best node to select. Halting.")
                break
        
        logger.info(f"Selected nodes: {selected_nodes}")
        logger.info(f"Total estimated influence: {total_influence:.2f}")
        
        return selected_nodes