import numpy as np
import pytest
from node_importance.diffusion import WeightedCascade
from node_importance.network import Network


# Fixtures for undirected graph components
@pytest.fixture
def undirected_component_1():
    """Fixture for the first component of the undirected graph."""
    return [
        (0, 1), (0, 2), (1, 2),
        (1, 3), (3, 4), (4, 5),
        (4, 6), (3, 7)
    ]


@pytest.fixture
def undirected_component_2():
    """Fixture for the second component of the undirected graph."""
    return [
        (8, 9), (9, 10),
        (9, 11), (11, 10)
    ]


@pytest.fixture
def undirected_graph(undirected_component_1, undirected_component_2):
    """Fixture for an undirected graph with two components."""
    G = Network(directed=False)
    for u, v in undirected_component_1 + undirected_component_2:
        G.add_edge(u, v)
    return G


@pytest.fixture
def weights_full(undirected_component_1, undirected_component_2):
    """Weights of 1 for all edges (both directions)."""
    weights = {}
    for u, v in undirected_component_1 + undirected_component_2:
        weights[(u, v)] = 1.0
        weights[(v, u)] = 1.0
    return weights


@pytest.fixture
def weights_zero(undirected_component_1, undirected_component_2):
    """Weights of 0 for all edges (both directions)."""
    weights = {}
    for u, v in undirected_component_1 + undirected_component_2:
        weights[(u, v)] = 0.0
        weights[(v, u)] = 0.0
    return weights


@pytest.fixture
def weights_half(undirected_component_1, undirected_component_2):
    """Weights of 0.5 for all edges (both directions)."""
    weights = {}
    for u, v in undirected_component_1 + undirected_component_2:
        weights[(u, v)] = 0.5
        weights[(v, u)] = 0.5
    return weights


@pytest.fixture
def weights_random(undirected_component_1, undirected_component_2):
    """Random weights in [0, 1] for all edges (both directions)."""
    rng = np.random.default_rng(0)
    weights = {}
    for u, v in undirected_component_1 + undirected_component_2:
        weights[(u, v)] = rng.random()
        weights[(v, u)] = rng.random()
    return weights


@pytest.fixture
def wiki_graph():
    """Fixture for Wiki vote graph."""
    G = Network()
    G.from_file('data/sample/Wiki-Vote.txt')
    return G


@pytest.fixture
def wiki_weights(wiki_graph):
    """Assign a small weight to each edge in the Wiki graph."""
    return {(u, v): 0.01 for u, v in wiki_graph.edges()}


def test_init(undirected_graph, weights_half):
    """Test if the WeightedCascade model is initialized correctly."""
    model = WeightedCascade(undirected_graph, weights=weights_half)
    assert model.weights == weights_half

    with pytest.raises(ValueError):
        WeightedCascade(undirected_graph, weights={(0, 1): -0.1})
    with pytest.raises(ValueError):
        WeightedCascade(undirected_graph, weights={(0, 1): 1.1})


def test_no_spread(undirected_graph, weights_zero):
    """Test spread with all weights set to 0."""
    model = WeightedCascade(undirected_graph, weights=weights_zero)
    out, history = model.run(undirected_graph, {0})
    assert out == {0}
    assert history == [{0}]


def test_full_spread(undirected_graph, weights_full):
    model = WeightedCascade(undirected_graph, weights=weights_full)

    # component 1
    out, history = model.run(undirected_graph, {0})
    assert out == {0, 1, 2, 3, 4, 5, 6, 7}
    assert history == [
        {0},
        {0, 1, 2},
        {0, 1, 2, 3},
        {0, 1, 2, 3, 4, 7},
        {0, 1, 2, 3, 4, 5, 6, 7},
    ]

    # component 2
    out2, history2 = model.run(undirected_graph, {8})
    assert out2 == {8, 9, 10, 11}
    assert history2 == [{8}, {8, 9}, {8, 9, 10, 11}]


def test_immunized(undirected_graph, weights_full):
    """Test spread with immunized nodes."""
    model = WeightedCascade(undirected_graph, weights=weights_full)

    # Immunize node 3 in component 1
    out, history = model.run(undirected_graph, {0}, {3})
    assert out == {0, 1, 2}
    assert history == [{0}, {0, 1, 2}]

    # Immunize node 9 in component 2
    out2, history2 = model.run(undirected_graph, {8}, {9})
    assert out2 == {8}
    assert history2 == [{8}]


def test_directed_spread():
    """Test that infection follows edge direction in directed graphs."""
    # Forward direction: 0 -> 1 -> 2
    G = Network(directed=True)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    weights = {(0, 1): 1.0, (1, 2): 1.0}
    model = WeightedCascade(G, weights=weights)
    out, history = model.run(G, {0})
    assert out == {0, 1, 2}
    assert history == [{0}, {0, 1}, {0, 1, 2}]

    # Reverse direction: 1 -> 0, 2 -> 1
    G_rev = Network(directed=True)
    G_rev.add_edge(1, 0)
    G_rev.add_edge(2, 1)
    weights_rev = {(1, 0): 1.0, (2, 1): 1.0}
    model_rev = WeightedCascade(G_rev, weights=weights_rev)
    out_rev, history_rev = model_rev.run(G_rev, {0})
    assert out_rev == {0}
    assert history_rev == [{0}]


def test_graph_with_no_edges():
    """Test with a graph with no edges."""
    G = Network()
    for node in [0, 1, 2]:
        G.add_node(node)

    model = WeightedCascade(G, weights={})
    out, history = model.run(G, {0})
    assert out == {0}
    assert history == [{0}]


def test_wiki_graph(wiki_graph, wiki_weights):
    """Test on a real-world graph (Wiki vote)."""
    model = WeightedCascade(wiki_graph, weights=wiki_weights)
    out, history = model.run(wiki_graph, {0})
    assert len(out) >= 1  # At least the initial node should be infected
    assert history[0] == {0}  # The first entry in history should be the initial infected set


def test_seed_reproducibility(undirected_graph, weights_random):
    """Ensure identical outcomes for runs with the same random seed."""
    model = WeightedCascade(undirected_graph, weights=weights_random)

    np.random.seed(42)
    out1, history1 = model.run(undirected_graph, {0})

    np.random.seed(42)
    out2, history2 = model.run(undirected_graph, {0})

    assert out1 == out2
    assert history1 == history2

    np.random.seed(24)
    out3, history3 = model.run(undirected_graph, {0})

    assert out3 != out1 or history3 != history1
