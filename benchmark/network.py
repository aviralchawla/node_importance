import cProfile
import pstats
import os
import random
import time

from node_importance.network import Network

def run_nodes(network, num_steps=5):
    """Retreive nodes from the network."""
    for _ in range(num_steps):
        nodes = network.nodes()
        node_labels = network.nodes_original()

def run_edges(network, num_steps=5):
    """Retreive edges from the network."""
    for _ in range(num_steps):
        edges = network.edges()

def run_neighbors(network, num_steps=5):
    """Retreive neighbors of nodes in the network."""
    for _ in range(num_steps):
        for node in network.nodes():
            neighbors = network.get_neighbors(node)

def main():
    """Main function to run the benchmark."""
    # --- Setup ---
    G = Network()
    G.from_file('../../data/sample/Wiki-Vote.txt')
    
    # --- Profiling ---
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_nodes(G, num_steps=100)
    # run_edges(G, num_steps=100)
    run_neighbors(G, num_steps=100)
    
    profiler.disable()
    
    # --- Save and Print Stats ---
    stats_file = f"profile_{time.time()}.prof"
    profiler.dump_stats(stats_file)
    
    print("--- cProfile Stats ---")
    p = pstats.Stats(stats_file)
    p.sort_stats("cumulative").print_stats(20)
    
    print(f"\nProfiling data saved to '{stats_file}'.")
    print("To visualize with snakeviz, run the following command in your terminal:")
    print(f"snakeviz {stats_file}")

if __name__ == "__main__":
    main()
