import numpy as np
import pytest
from node_importance.diffusion import IndependentCascade
from node_importance.network import Network

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
def wiki_graph():
    """Fixture for Wiki vote graph."""
    G = Network()
    G.from_file('data/sample/Wiki-Vote.txt')
    return G

def test_init():
    """Test if the IndependentCascade model is initialized correctly."""
    model = IndependentCascade(p=0.5)
    assert model.p == 0.5

    # Test with invalid probability
    with pytest.raises(ValueError):
        IndependentCascade(p=-0.1)
    with pytest.raises(ValueError):
        IndependentCascade(p=1.1)

def test_no_spread(undirected_graph):
    """Test spread with probability 0."""
    model = IndependentCascade(p=0)
    out, history = model.run(undirected_graph, {0})
    assert out == {0}
    assert history == [{0}]

def test_full_spread(undirected_graph):
    model = IndependentCascade(p=1)

    # component 1
    out, history = model.run(undirected_graph, {0})
    assert out == {0, 1, 2, 3, 4, 5, 6, 7}
    assert history == [{0}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4, 7}, {0, 1, 2, 3, 4, 5, 6, 7}]

    # component 2
    out, history = model.run(undirected_graph, {8})
    assert out == {8, 9, 10, 11}
    assert history == [{8}, {8, 9}, {8, 9, 10, 11}]

def test_immunized(undirected_graph):
    """Test spread with immunized nodes."""
    model = IndependentCascade(p=1)

    # Immunize node 3 in component 1
    out, history = model.run(undirected_graph, {0}, {3})
    assert out == {0, 1, 2}
    assert history == [{0}, {0, 1, 2}]

    # Immunize node 9 in component 2
    out, history = model.run(undirected_graph, {8}, {9})
    assert out == {8}
    assert history == [{8}]


def test_directed_spread():
    """Test that infection follows edge direction in directed graphs."""
    model = IndependentCascade(p=1)

    # Forward direction: 0 -> 1 -> 2
    G = Network(directed=True)
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    out, history = model.run(G, {0})
    assert out == {0, 1, 2}
    assert history == [{0}, {0, 1}, {0, 1, 2}]

    # Reverse direction: 1 -> 0, 2 -> 1
    G_rev = Network(directed=True)
    G_rev.add_edge(1, 0)
    G_rev.add_edge(2, 1)

    out_rev, history_rev = model.run(G_rev, {0})
    assert out_rev == {0}
    assert history_rev == [{0}]

def test_graph_with_no_edges():
    """Test with a graph with no edges."""
    G = Network()
    for node in [0, 1, 2]:
        G.add_node(node)

    model = IndependentCascade(p=0.5)
    out, history = model.run(G, {0})
    assert out == {0}
    assert history == [{0}]

def test_wiki_graph(wiki_graph):
    """Test on a real-world graph (Wiki vote)."""
    model = IndependentCascade(p=0.01)
    out, history = model.run(wiki_graph, {0})
    assert len(out) >= 1  # At least the initial node should be infected
    assert history[0] == {0}  # The first entry in history should be the initial infected set


def test_seed_reproducibility(undirected_graph):
    """Ensure identical outcomes for runs with the same random seed."""
    model = IndependentCascade(p=0.5)

    np.random.seed(42)
    out1, history1 = model.run(undirected_graph, {0})

    np.random.seed(42)
    out2, history2 = model.run(undirected_graph, {0})

    assert out1 == out2
    assert history1 == history2

    np.random.seed(24)
    out3, history3 = model.run(undirected_graph, {0})

    assert out3 != out1 or history3 != history1

def test_independent_trials():
    pass
