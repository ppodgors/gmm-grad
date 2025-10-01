import os
import pickle
import numpy as np

def save_results(results, filepath):
    """Saves the results dictionary to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults successfully saved to '{filepath}'")

def load_results_if_exist(filepath):
    """Loads results from a pickle file if it exists."""
    if os.path.exists(filepath):
        print(f"Found existing results file. Loading from '{filepath}'")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def print_summary(results):
    """Prints a summary of mean ARI and NMI scores."""
    print("\n--- Final Mean Scores (averaged over all worlds and repetitions) ---")
    for method in results:
        if method == 'config': continue
        mean_ari = np.nanmean(results[method]['ari'])
        mean_nmi = np.nanmean(results[method]['nmi'])
        print(f"{method:<10}: ARI = {mean_ari:.4f}, NMI = {mean_nmi:.4f}")