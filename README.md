# Node Importance

Node Importance is a Python library for experimenting and prototyping node importance problems in networks. The repository provides a wrapper around `networkx` and `graph-tool` with a standardized diffusion models. You can use it to simulate contagion processes, run influence maximization algorithms, and benchmark different approaches.

## Features

- Efficient `Network` graph wrapper with cached neighbour lookups.
- Diffusion models for simulating spread:
  - `IndependentCascade` – classic independent cascade contagion model.
  - `WeightedCascade` – probabilistic spread based on per–edge weights.
- Influence maximization algorithms:
  - `Greedy` – iterative Monte‑Carlo selection.
  - `Genetic` – evolutionary search over seed sets.
  - `NeighborHop Genetic` - genetic algorithm with neighbor hops.
- Benchmark scripts and example notebooks to get started quickly.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/aviralchawla/node_importance.git
cd node_importance
pip install -r requirements.txt
pip install -e ./
```

## Usage

```python
from node_importance.network import Network
from node_importance.diffusion import IndependentCascade
from node_importance.influence_maximization import Greedy

# Build a network from an edge list
G = Network()
G.from_file("data/sample/Wiki-Vote.txt")

# Run a diffusion simulation
model = IndependentCascade(p=0.1)
infected, history = model.run(G, {0})

# Solve influence maximization
greedy = Greedy(diffusion_model=model, num_samples=10)
seed_set = greedy.fit(G, k=5)
print(seed_set)
```

## Project Structure

- `node_importance/` – core library code.
- `tests/` – pytest test suite for diffusion models.
- `data/sample/` – small sample graphs used by tests and examples.
- `examples/` – Jupyter notebooks demonstrating usage.
- `benchmark/` – scripts for profiling components.

## Testing

Run the test suite with:

```bash
pytest
```

## Contributing

Pull requests are welcome. Please run the tests and ensure lint checks pass before submitting.

## Upcoming Features
- [ ] Support for diffusion models (Linear Threshold, etc.)
- [ ] More influence maximization algorithms (CELF, Simulated Annealing, etc.)
- [ ] Suite of targeted immunization & sentinel surveillance algorithms.
- [ ] Publish to PyPI for easier installation.

## License

Licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

