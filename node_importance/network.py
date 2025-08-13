import numpy as np
import warnings
import logging
from typing import List, Tuple, Dict, Type
from abc import ABC, abstractmethod

try:
    import graph_tool
    GRAPH_TOOL_AVAILABLE = True
except ImportError:
    GRAPH_TOOL_AVAILABLE = False

logger = logging.getLogger(__name__)

BACKEND_REGISTRY: Dict[str, Type['_NetworkBackend']] = {}

def register_backend(name: str):
    """
    A decorator to register a backend class in the BACKEND_REGISTRY.
    It checks for dependencies before registering.
    """
    def decorator(cls: Type['_NetworkBackend']) -> Type['_NetworkBackend']:
        if name == "graph_tool" and not GRAPH_TOOL_AVAILABLE:
            logger.debug("graph-tool not available, skipping its backend registration.")
        else:
            BACKEND_REGISTRY[name] = cls
            logger.debug(f"Registered backend: '{name}'")
        return cls
    return decorator

######## ABSTRACT BASE CLASS ########

class _NetworkBackend(ABC):
    """Abstract base class defining the common interface for all backends."""
    @abstractmethod
    def from_file(self, file_path: str, directed: bool):
        pass

    @abstractmethod
    def from_graph(self, graph):
        pass

    @property
    @abstractmethod
    def graph(self):
        pass

    @abstractmethod
    def nodes(self) -> List[int]:
        pass

    @abstractmethod
    def edges(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def add_edge(self, u: int, v: int):
        pass

    @abstractmethod
    def add_node(self, node: int):
        pass

    @abstractmethod
    def get_degree(self, node: int) -> int:
        pass

    @abstractmethod
    def get_in_degree(self, node: int) -> int:
        pass

    @abstractmethod
    def get_out_degree(self, node: int) -> int:
        pass

    @abstractmethod
    def get_neighbors(self, node: int) -> List[int]:
        pass

    @abstractmethod
    def number_of_nodes(self) -> int:
        pass

    @abstractmethod
    def number_of_edges(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


# --- Backend Implementations ---

@register_backend("networkx")
class _NetworkXBackend(_NetworkBackend):
    def __init__(self, directed: bool = False):
        import networkx as nx
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
        import networkx as nx
        graph_type = nx.DiGraph if directed else nx.Graph
        self.directed = directed
        logger.info(f"Loading graph from {file_path} using NetworkX parser...")
        try:
            nx_graph = nx.read_edgelist(
                file_path,
                create_using=graph_type,
                nodetype=str,
            )
            nx_graph = nx.relabel.convert_node_labels_to_integers(nx_graph)
            self._graph = nx_graph
            self._num_nodes = nx_graph.number_of_nodes()
            self._num_edges = nx_graph.number_of_edges()
            self._cache_neighbors()
        except Exception as e:
            raise IOError(f"Error loading file {file_path} with NetworkX: {e}") from e

    def from_graph(self, graph):
        import networkx as nx
        self._graph = graph
        self._num_nodes = graph.number_of_nodes()
        self._num_edges = graph.number_of_edges()
        self.directed = graph.is_directed()
        self._cache_neighbors()

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
    def graph(self):
        return self._graph

    def nodes(self) -> List[int]:
        """ Returns a list of nodes (int) """
        return self._graph.nodes()

    def edges(self) -> List[Tuple[int, int]]:
        """ Returns a list of edges as tuples (u, v) """
        return list(self._graph.edges())

    def add_edge(self, u: int, v: int):
        self._graph.add_edge(u, v)
        self._num_nodes = self._graph.number_of_nodes()
        self._num_edges = self._graph.number_of_edges()
        self._cache_neighbors() # Recache on modification

    def add_node(self, node: int):
        self._graph.add_node(node)
        self._num_nodes = self._graph.number_of_nodes()
        self._cache_neighbors() # Recache on modification

    def get_degree(self, node: int) -> int:
        if self.directed:
            return self._graph.out_degree(node)
        return self._graph.degree(node)

    def get_in_degree(self, node: int) -> int:
        if self.directed:
            return self._graph.in_degree(node)
        raise ValueError("In-degree is only defined for directed graphs.")

    def get_out_degree(self, node: int) -> int:
        if self.directed:
            return self._graph.out_degree(node)
        raise ValueError("Out-degree is only defined for directed graphs.")

    def get_neighbors(self, node: int) -> List[int]:
        return self._neighbors_cache[node]

    def number_of_nodes(self) -> int:
        return self._num_nodes

    def number_of_edges(self) -> int:
        return self._num_edges

    def __len__(self) -> int:
        return self._num_nodes

    def __str__(self) -> str:
        return (f"Network(Nodes={self.number_of_nodes()}, Edges={self.number_of_edges()}, "
                f"Directed={self.directed}, Format=NetworkX)")

@register_backend("graph_tool")
class _GraphToolBackend(_NetworkBackend):
    def __init__(self, directed: bool = False):
        import graph_tool.all as gt
        self.directed = directed
        self._graph = gt.Graph(directed=self.directed)
        self._neighbors_cache = []

    def from_file(self, file_path: str, directed: bool = False):
        """
        Loads a graph from an edge list file using the graph-tool parser.
        Parameters:
        ----------
        file_path : str
            Path to the edge list file.
        directed : bool, optional
            If True, the graph is treated as directed; otherwise, it is treated as undirected.
        """
        import graph_tool.all as gt
        self.directed = directed
        logger.info(f"Loading graph from {file_path} using graph-tool parser...")
        try:
            # graph-tool can load from edgelist but it's a bit different
            self._graph = gt.load_graph_from_csv(file_path, directed=directed, csv_options={'delimiter': ' '})
            self._cache_neighbors()
        except Exception as e:
            raise IOError(f"Error loading file {file_path} with graph-tool: {e}") from e

    def from_graph(self, graph):
        self._graph = graph
        self.directed = graph.is_directed()
        self._cache_neighbors()

    def _cache_neighbors(self):
        """
        Caches the neighbors of each node for quick access.
        This is useful when we repeatedly need neighbors for contagion spreads.
        """
        logger.debug("Caching neighbors for all nodes...")
        num_nodes = self.number_of_nodes()
        if num_nodes == 0:
            self._neighbors_cache = []
            return
        neighbors = np.empty(num_nodes, dtype=object)
        for node in self._graph.vertices():
            neighbors[int(node)] = [int(neighbor) for neighbor in node.out_neighbors()]
        self._neighbors_cache = neighbors

    @property
    def graph(self):
        return self._graph

    def nodes(self) -> List[int]:
        return [int(v) for v in self._graph.vertices()]

    def edges(self) -> List[Tuple[int, int]]:
        return [(int(e.source()), int(e.target())) for e in self._graph.edges()]

    def add_edge(self, u: int, v: int):
        self._graph.add_edge(self._graph.vertex(u), self._graph.vertex(v))
        self._cache_neighbors() # Recache on modification

    def add_node(self, node: int):
        # Add vertices up to the desired node index if they don't exist
        num_to_add = (node + 1) - self.number_of_nodes()
        if num_to_add > 0:
            self._graph.add_vertex(num_to_add)
        self._cache_neighbors() # Recache on modification

    def get_degree(self, node: int) -> int:
        return self._graph.vertex(node).out_degree()

    def get_in_degree(self, node: int) -> int:
        if self.directed:
            return self._graph.vertex(node).in_degree()
        raise ValueError("In-degree is only defined for directed graphs.")

    def get_out_degree(self, node: int) -> int:
        if self.directed:
            return self._graph.vertex(node).out_degree()
        raise ValueError("Out-degree is only defined for directed graphs.")

    def get_neighbors(self, node: int) -> List[int]:
        return self._neighbors_cache[node]

    def number_of_nodes(self) -> int:
        return self._graph.num_vertices()

    def number_of_edges(self) -> int:
        return self._graph.num_edges()

    def __len__(self) -> int:
        return self.number_of_nodes()

    def __str__(self) -> str:
        return (f"Network(Nodes={self.number_of_nodes()}, Edges={self.number_of_edges()}, "
                f"Directed={self.directed}, Format=graph-tool)")


######## UNIFIED WRAPPER ########

class Network:
    """
    A unified network interface that dynamically proxies calls to a selected
    backend (e.g., NetworkX or graph-tool). 
    """
    _backend: _NetworkBackend

    def __init__(self, graph=None, file_path: str = None, directed: bool = False, backend: str = "auto"):
        """
        Initializes the Network object using a dynamically selected backend.

        Args:
            graph: An existing graph object from a supported library.
            file_path: Path to an edgelist file to load.
            directed: Whether the graph is directed.
            backend: The backend to use: "networkx", "graph_tool", or "auto".
                     "auto" prefers graph-tool if available, otherwise falls back to networkx.
        """
        if backend == "auto":
            chosen_backend_name = "graph_tool" if "graph_tool" in BACKEND_REGISTRY else "networkx"
        elif backend not in BACKEND_REGISTRY:
            raise ValueError(f"Unknown backend '{backend}'. Available: {list(BACKEND_REGISTRY.keys())}")
        else:
            chosen_backend_name = backend

        backend_class = BACKEND_REGISTRY[chosen_backend_name]
        self._backend = backend_class(directed=directed)
        logger.info(f"Using '{chosen_backend_name}' backend.")

        # Initial loading logic is delegated to the chosen backend
        if file_path:
            self._backend.from_file(file_path, directed)
        elif graph is not None:
            self._backend.from_graph(graph)

    def __getattr__(self, name: str):
        """
        Dynamically delegates attribute and method calls to the backend instance.
        This eliminates all repetitive proxy methods.
        """
        try:
            return getattr(self._backend, name)
        except AttributeError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. It was also not found "
                f"on the '{type(self._backend).__name__}' backend."
            ) from e

    def __len__(self) -> int:
        """Returns the number of nodes in the graph."""
        return len(self._backend)

    def __str__(self) -> str:
        """Returns a string representation of the graph from its backend."""
        return str(self._backend)
