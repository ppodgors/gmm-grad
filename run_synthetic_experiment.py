import numpy as np
import time
import sys
import os
import argparse
import random
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.regularized_gmm import RegularizedGMM
from src.data_generator import generate_world_parameters, generate_trial_dataset

def run_single_trial(X_data, y_true, k_components, trial_seed):
    """
    Runs all models on a single dataset, setting seeds and parameters
    """
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    torch.manual_seed(trial_seed)

    metrics = {}

    # --- 1. Standard scikit-learn GMM ---
    t0_gmm = time.perf_counter()
    gmm = GaussianMixture(n_components=k_components, n_init=10) 
    labels_gmm = gmm.fit_predict(X_data)
    metrics['time_gmm'] = time.perf_counter() - t0_gmm
    metrics['ari_gmm'] = adjusted_rand_score(y_true, labels_gmm)
    metrics['nmi_gmm'] = normalized_mutual_info_score(y_true, labels_gmm)

    # --- 2. Regularized GMM with Gradient method ---
    t0_grad = time.perf_counter()
    model_grad = RegularizedGMM(k_components=k_components, eta_method='grad')
    model_grad.fit(X_data, seed=trial_seed) 
    labels_grad = model_grad.predict(X_data)
    metrics['time_grad'] = time.perf_counter() - t0_grad
    metrics['ari_grad'] = adjusted_rand_score(y_true, labels_grad)
    metrics['nmi_grad'] = normalized_mutual_info_score(y_true, labels_grad)

    # --- 3. Regularized GMM with Grid Search (GMM-GS) method ---
    t0_gs = time.perf_counter()
    model_gs = RegularizedGMM(k_components=k_components, eta_method='gs')
    model_gs.fit(X_data, seed=trial_seed)
    labels_gs = model_gs.predict(X_data)
    metrics['time_gs'] = time.perf_counter() - t0_gs
    metrics['ari_gs'] = adjusted_rand_score(y_true, labels_gs)
    metrics['nmi_gs'] = normalized_mutual_info_score(y_true, labels_gs)

    return metrics

def main(args):
    D_VALUES = range(args.d_start, args.d_stop, args.d_step)
    N_SAMPLES_PER_COMPONENT = args.n_samples
    NUM_WORLDS = args.num_worlds
    NUM_REPLICATIONS = args.num_reps
    K_COMPONENTS = 3

    num_total_trials = NUM_WORLDS * NUM_REPLICATIONS
    num_d_steps = len(D_VALUES)
    
    results = {
        'ari_grad': np.zeros((num_total_trials, num_d_steps)), 'ari_gs': np.zeros((num_total_trials, num_d_steps)), 'ari_gmm': np.zeros((num_total_trials, num_d_steps)),
        'nmi_grad': np.zeros((num_total_trials, num_d_steps)), 'nmi_gs': np.zeros((num_total_trials, num_d_steps)), 'nmi_gmm': np.zeros((num_total_trials, num_d_steps)),
        'time_grad': np.zeros((num_total_trials, num_d_steps)), 'time_gs': np.zeros((num_total_trials, num_d_steps)), 'time_gmm': np.zeros((num_total_trials, num_d_steps)),
    }

    for d_idx, d_features in enumerate(D_VALUES):
        for world_idx in range(NUM_WORLDS):
            world_seed = world_idx * 13
            np.random.seed(world_seed)
            random.seed(world_seed)
            torch.manual_seed(world_seed)

            true_mus, true_covs = generate_world_parameters(d_features, K_COMPONENTS, world_seed)
            
            for rep_idx in range(NUM_REPLICATIONS):
                trial_seed = 10000 * world_idx + rep_idx
                trial_flat_idx = world_idx * NUM_REPLICATIONS + rep_idx
                
                np.random.seed(trial_seed)
                random.seed(trial_seed)
                torch.manual_seed(trial_seed)

                X_data, y_true = generate_trial_dataset(N_SAMPLES_PER_COMPONENT, true_mus, true_covs, trial_seed)
                trial_metrics = run_single_trial(X_data, y_true, K_COMPONENTS, trial_seed)
                
                for key, value in trial_metrics.items():
                    results[key][trial_flat_idx, d_idx] = value
                
                progress_info = (f"n={N_SAMPLES_PER_COMPONENT}, d={d_features}, world={world_idx+1}/{NUM_WORLDS}, rep={rep_idx+1}/{NUM_REPLICATIONS} | "f"ARI GS: {results['ari_gs'][trial_flat_idx, d_idx]:.3f}, "f"ARI Grad: {results['ari_grad'][trial_flat_idx, d_idx]:.3f}")
                
                sys.stdout.write(f"\r{progress_info}")
                sys.stdout.flush()

    print("\nExperiment finished.")
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    results_filename = os.path.join(output_dir, f"results_n{N_SAMPLES_PER_COMPONENT}.npz")
    np.savez(results_filename, d_values=list(D_VALUES), n_value=N_SAMPLES_PER_COMPONENT, **results)
    print(f"Results saved to {results_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the synthetic data experiment for Regularized GMM.")
    parser.add_argument('--n_samples', type=int, required=True, help="Number of samples per component.")
    parser.add_argument('--num_worlds', type=int, default=25, help="Number of worlds to generate.")
    parser.add_argument('--num_reps', type=int, default=4, help="Number of replications per world.")
    parser.add_argument('--d_start', type=int, default=20, help="Starting number of features (d).")
    parser.add_argument('--d_stop', type=int, default=101, help="Ending number of features (d), exclusive.")
    parser.add_argument('--d_step', type=int, default=10, help="Step for the number of features range.")
    args = parser.parse_args()
    main(args)


               
