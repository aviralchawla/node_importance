import numpy as np
import networkx as nx
import warnings
import logging
from typing import List, Tuple

# Set up a logger for this module
logger = logging.getLogger(__name__)

class Network:
    """
    A wrapper class for graph operations using a SciPy sparse matrix for efficiency.
    """
    def __init__(self, directed: bool = False):
        """
        Initialize the Network object.
        """
        self.directed = directed
        self._neighbors_cache = []
        self._num_nodes = 0
        self._num_edges = 0
        self._graph = nx.DiGraph() if directed else nx.Graph()

    def from_file(self, file_path: str, directed: bool = False):
        """
        Loads a graph from an edge list file using the NetworkX parser.

        Parameters:
        ----------
        file_path : str
            Path to the edge list file.
        delimiter : str, optional
            Delimiter used in the edge list file. If None, defaults to whitespace.
        directed : bool, optional
            If True, the graph is treated as directed; otherwise, it is treated as undirected.
            Defaults to False.
        """
        
        graph_type = nx.DiGraph if directed else nx.Graph
        self.directed = directed
        
        logger.info(f"Loading graph from {file_path} using NetworkX parser...")
        
        try:
            nx_graph = nx.read_edgelist(
                file_path,
                create_using=graph_type,
                nodetype=str,
            )
            nx_graph = nx.relabel.convert_node_labels_to_integers(
                nx_graph
            )
            self._graph = nx_graph

            self._num_nodes = nx_graph.number_of_nodes()
            self._num_edges = nx_graph.number_of_edges()

            self._cache_neighbors()

        except Exception as e:
            raise IOError(f"Error loading file {file_path} with NetworkX: {e}")

    def _cache_neighbors(self):
        """
        Caches the neighbors of each node for quick access.
        This is useful when we repeatedly need neighbors for contagion spreads.
        """
        logger.debug("Caching neighbors for all nodes...")
        neighbors = np.empty(self._num_nodes, dtype=object)
        for node in range(self._num_nodes):
            neighbors[node] = list(self._graph.neighbors(node))
        self._neighbors_cache = neighbors

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def nodes(self) -> List[int]:
        """Returns the list of int nodes""" 
        return self._graph.nodes()
    
    def edges(self) -> List[Tuple[int, int]]:
        """Returns the list of edges as tuples of node indices."""
        return list(self._graph.edges())

    def add_edge(self, u: int, v: int):
        """
        Adds an edge between nodes u and v to the graph.
        If the graph is directed, the edge is directed from u to v.
        """
        if not self._graph.has_node(u):
            self._graph.add_node(u)
            self._num_nodes += 1
        if not self._graph.has_node(v):
            self._graph.add_node(v)
            self._num_nodes += 1
        
        self._graph.add_edge(u, v)
        self._num_edges += 1
        
        # Update neighbors cache
        self._neighbors_cache = []
        self._cache_neighbors()
    
    def add_node(self, node: int):
        """
        Adds a single node to the graph.
        """
        if not self._graph.has_node(node):
            self._graph.add_node(node)
            self._num_nodes += 1
            
            # Update neighbors cache
            self._neighbors_cache = []
            self._cache_neighbors()

    def get_degree(self, node: int) -> int:
        """ Returns the degree of a node, if the graph is directed, returns the out-degree."""
        if self.directed:
            return self._graph.out_degree(node)
        else:
            return self._graph.degree(node)
    
    def get_in_degree(self, node: int) -> int:
        """ Returns the in-degree of a node, if the graph is directed."""
        if self.directed:
            return self._graph.in_degree(node)
        else:
            raise ValueError("In-degree is only defined for directed graphs.")
    
    def get_out_degree(self, node: int) -> int:
        """ Returns the out-degree of a node, if the graph is directed."""
        if self.directed:
            return self._graph.out_degree(node)
        else:
            raise ValueError("Out-degree is only defined for directed graphs.")

    def get_neighbors(self, node: int) -> List[int]:
        """Returns the neighbors of a node from the cache."""
        return self._neighbors_cache[node]

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._num_nodes

    def number_of_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return self._num_edges

    def __len__(self) -> int:
        return self._num_nodes

    def __str__(self) -> str:
        """Formats the graph information as a string."""
        return (f"Network(Nodes={self.number_of_nodes()}, Edges={self.number_of_edges()}, "
                f"Directed={self.directed}, Format={type(self._graph).__name__})")
