import pytest

from node_importance.network import Network


class TestUndirectedNetwork:
    """Tests for functionality of undirected Network."""

    @pytest.fixture
    def network(self):
        return Network()

    def test_initial_state(self, network):
        assert not network.directed
        assert network.graph is not None
        assert network.number_of_nodes() == 0
        assert network.number_of_edges() == 0
        assert list(network.nodes()) == []
        assert network.edges() == []

    def test_add_node_and_edge(self, network):
        network.add_node(0)
        assert network.number_of_nodes() == 1
        assert network.get_neighbors(0) == []

        network.add_edge(0, 1)
        assert network.number_of_nodes() == 2
        assert network.number_of_edges() == 1
        assert list(network.nodes()) == [0, 1]
        assert set(network.edges()) == {(0, 1)}
        assert network.get_neighbors(0) == [1]
        assert network.get_neighbors(1) == [0]

    def test_degree_methods(self, network):
        network.add_edge(0, 1)
        network.add_edge(0, 2)

        assert network.get_degree(0) == 2
        with pytest.raises(ValueError):
            network.get_in_degree(0)
        with pytest.raises(ValueError):
            network.get_out_degree(0)

    def test_len_and_str(self, network):
        network.add_edge(0, 1)
        info = str(network)
        assert len(network) == 2
        assert "Nodes=2" in info
        assert "Edges=1" in info
        assert "Directed=False" in info


class TestDirectedNetwork:
    """Tests for functionality of directed Network."""

    @pytest.fixture
    def network(self):
        return Network(directed=True)

    def test_add_edges_and_degrees(self, network):
        network.add_edge(0, 1)
        network.add_edge(1, 2)

        assert network.directed
        assert network.get_neighbors(0) == [1]
        assert network.get_neighbors(1) == [2]
        assert network.get_neighbors(2) == []

        assert network.get_degree(1) == 1
        assert network.get_in_degree(1) == 1
        assert network.get_out_degree(1) == 1

    def test_len_and_str(self, network):
        network.add_edge(0, 1)
        info = str(network)
        assert len(network) == 2
        assert "Nodes=2" in info
        assert "Edges=1" in info
        assert "Directed=True" in info


class TestFileLoading:
    """Tests related to loading graphs from files."""

    def test_from_file_undirected(self, tmp_path):
        edge_file = tmp_path / "edges.txt"
        edge_file.write_text("a b\nb c\n")

        net = Network()
        net.from_file(str(edge_file))

        assert set(net.nodes()) == {0, 1, 2}
        assert set(net.edges()) == {(0, 1), (1, 2)}
        assert set(net.get_neighbors(1)) == {0, 2}
        assert not net.directed

    def test_from_file_directed(self, tmp_path):
        edge_file = tmp_path / "edges.txt"
        edge_file.write_text("0 1\n1 0\n")

        net = Network()
        net.from_file(str(edge_file), directed=True)

        assert net.directed
        assert set(net.edges()) == {(0, 1), (1, 0)}
        assert net.get_neighbors(0) == [1]
        assert net.get_in_degree(0) == 1
        assert net.get_out_degree(0) == 1

    def test_from_file_missing(self):
        net = Network()
        with pytest.raises(IOError):
            net.from_file("nonexistent.txt")
