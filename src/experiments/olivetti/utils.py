import os
import pickle

def save_experiment_results(results_data, output_dir, filename):
    """Creates the output directory and saves results data to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"\n3. Results saved to: {filepath}")