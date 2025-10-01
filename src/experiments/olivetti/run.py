import random
import numpy as np
import torch
import time

from .data import prepare_olivetti_data
from .loop import execute_experiment_loop
from .utils import save_experiment_results

def run_olivetti_experiment(n_rep, main_seed, num_subjects, output_dir):

    start_time = time.time()
    
    random.seed(main_seed)
    np.random.seed(main_seed)
    torch.manual_seed(main_seed)

    # 1. Prepare data
    person_ids = list(range(num_subjects))
    pool_size = 4 
    X, y_true, n_classes = prepare_olivetti_data(person_ids, pool_size)

    # 2. Execute the main experiment loop
    loop_results = execute_experiment_loop(X, y_true, n_classes, n_rep, main_seed)

    total_time = time.time() - start_time

    print("\n-------------------------------------------")
    print("Final Mean Scores Across All Repetitions:")
    print("-------------------------------------------")
    print(f"GMM-grad: ARI = {loop_results['ari_grad'].mean():.4f}, NMI = {loop_results['nmi_grad'].mean():.4f}")
    print(f"GMM-GS:   ARI = {loop_results['ari_gs'].mean():.4f}, NMI = {loop_results['nmi_gs'].mean():.4f}")
    print(f"GMM-STD:  ARI = {loop_results['ari_gmm'].mean():.4f}, NMI = {loop_results['nmi_gmm'].mean():.4f}")
    print("-------------------------------------------")

    # 3. Package and save the results
    final_results_package = {
        **loop_results,
        'person_ids': person_ids,
        'n_rep': n_rep,
        'main_seed': main_seed,
        'pool_block': pool_size,
        'feature_dim': X.shape[1],
        'total_time_sec': total_time
    }

    save_experiment_results(
        results_data=final_results_package,
        output_dir=output_dir,
        filename='results.pkl'
    )
    
    print(f"\nTotal experiment execution time: {total_time:.2f} s")