import os
from .data import prepare_newsgroups_data
from .loop import execute_experiment_loop
from .utils import save_results, load_results_if_exist, print_summary

def run_newsgroups_experiment(n_worlds, n_reps, data_dir, categories, samples_per_cat):
    
    # STEP 1: Ensure data exists, generate if necessary
    print("--- Data Preparation Step ---")
    prepare_newsgroups_data(
        n_worlds=n_worlds,
        categories=categories,
        n_per_cat=samples_per_cat,
        output_dir=data_dir
    )
    
    # STEP 2: Run the experiment
    print("\n--- Experiment Execution Step ---")
    results_file = os.path.join(data_dir, f'results_w{n_worlds}_r{n_reps}.pkl')

    results = load_results_if_exist(results_file)
    if results is None:
        print("No existing results file found. Starting a new experiment.")
        results = execute_experiment_loop(data_dir, n_worlds, n_reps)
        
        results['config'] = {
            'categories': categories,
            'n_samples_per_cat': samples_per_cat,
            'n_worlds': n_worlds,
            'n_reps': n_reps
        }
        save_results(results, results_file)
    
    print_summary(results)