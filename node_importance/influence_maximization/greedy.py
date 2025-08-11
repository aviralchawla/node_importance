import logging
from typing import Set
from tqdm import tqdm
import os

from node_importance.diffusion import DiffusionModel
from node_importance.network import Network

logger = logging.getLogger(__name__)


class Greedy:
    """
    The general Greedy algorithm maximizes the influence by iteratively selecting the node with the highest potential for
    additional influence until the set is full. First, we calculate R({v}), ∀v ∈ G. From this, we select the node, n1,
    with the highest measured influence. Next, we calculate R({n1, v}), ∀v ∈ G − {n1}, and then pick the set with the
    highest influence, {n1, n2}. We repeat this process till our set is of size k.

    The algorithm was proposed by Kempe et al. in "Maximizing the Spread of Influence through a Social Network" (2003).
    """

    def __init__(self, diffusion_model: DiffusionModel, num_samples: int = 10):
        """
        Initialize the Greedy algorithm for influence maximization.

        Parameters:
        ----------
        diffusion_model : DiffusionModel
            The contagion model to be used for influence maximization.
        num_samples : int
            Number of samples to use for estimating the influence of nodes. Default is 10.
        """
        self.diffusion_model = diffusion_model
        self.num_samples = num_samples

    def _run_single(network: Network, diffusion_model: DiffusionModel, node: int, 
                    selected_ndoes: Set[int], _unused=None) -> int:
        """Run one Monte-Carlo simulation and return outbreak size."""
        infected = {node} | selected_ndoes
        spread, _ = diffusion_model(
            network,
            infected
        )
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
        # Greedy relies on repeated MC sampling to estimate influence
        results = []
        # great potential for parallelization here
        for _ in range(self.num_samples):
            result = self._run_single(network, self.diffusion_model, node, selected_nodes)
            results.append(result)

        influence = sum(results) / self.num_samples

        return influence
        

    def fit(self, network: Network, k: int) -> Set[int]:
        """
        Find top-k nodes for influence maximization using the Greedy algorithm.

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
        selected_nodes = set()
        remaining_nodes = set(network.nodes())
        best_influence = None

        for i in range(k):
            logger.info(f"Selecting node {i + 1}/{k}...")

            best_node = None

            # Note: tqdm prints to stderr by default, which is fine.
            for node in tqdm(remaining_nodes, desc=f"Scanning for node {i + 1}"):
                influence = self._estimate_influence(network, node, selected_nodes)
                if best_influence is None or influence > best_influence:
                    best_influence = influence
                    best_node = node

            if best_node is not None:
                logger.info(f"Selected node {best_node} with influence {best_influence:.2f}")
                selected_nodes.add(best_node)
                remaining_nodes.remove(best_node)
            else:
                logger.warning("Could not find a best node to select. Halting.")
                break

        logger.info(f"Selected nodes: {selected_nodes}")
        logger.info(f"Final influence: {best_influence:.2f}")

        return selected_nodes
