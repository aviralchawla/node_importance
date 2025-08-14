import cProfile
import pstats
import os
import random
import time

from node_importance.network import Network
from node_importance.diffusion import IndependentCascade

def run_simulation(network, contagion_model, num_steps=5, max_outbreak=10):
    """Runs the contagion simulation for a number of steps."""

    for outbreak_size in range(2, max_outbreak + 1):
        initial_infected = random.sample(list(network.nodes()), outbreak_size)
        for _ in range(num_steps):
            out, history = contagion_model.run(network, set(initial_infected))

def main():
    """Main function to run the benchmark."""
    # --- Setup ---
    G = Network()
    G.from_file('../data/sample/Wiki-Vote.txt')
    contagion_model = IndependentCascade(p=0.01)
    
    # --- Profiling ---
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_simulation(G, contagion_model, num_steps=5_00)
    
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
