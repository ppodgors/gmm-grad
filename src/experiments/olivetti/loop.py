import time
import numpy as np
import random
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.regularized_gmm import RegularizedGMM 

def execute_experiment_loop(X, y_true, n_classes, n_rep, main_seed):

    print(f"\n2. Starting experiment loop ({n_rep} repetitions)...")

    results = {
        'ari_grad': np.zeros(n_rep), 'nmi_grad': np.zeros(n_rep), 'time_grad': np.zeros(n_rep),
        'ari_gs': np.zeros(n_rep), 'nmi_gs': np.zeros(n_rep), 'time_gs': np.zeros(n_rep),
        'ari_gmm': np.zeros(n_rep), 'nmi_gmm': np.zeros(n_rep), 'time_gmm': np.zeros(n_rep),
    }

    seeds = np.arange(n_rep)
    
    for i, seed in enumerate(seeds):
        s = seed * 41 + main_seed
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

        # GMM-grad
        t0 = time.time()
        gmm_grad = RegularizedGMM(k_components=n_classes, eta_method='grad')
        gmm_grad.fit(X, seed=s)
        lab_g = gmm_grad.predict(X)
        results['time_grad'][i] = time.time() - t0
        results['ari_grad'][i] = adjusted_rand_score(y_true, lab_g)
        results['nmi_grad'][i] = normalized_mutual_info_score(y_true, lab_g)

        # GMM-GS
        t0 = time.time()
        gmm_gs = RegularizedGMM(k_components=n_classes, eta_method='gs')
        gmm_gs.fit(X, seed=s)
        lab_c = gmm_gs.predict(X)
        results['time_gs'][i] = time.time() - t0
        results['ari_gs'][i] = adjusted_rand_score(y_true, lab_c)
        results['nmi_gs'][i] = normalized_mutual_info_score(y_true, lab_c)

        # Standard GMM
        t0 = time.time()
        gmm = GaussianMixture(n_components=n_classes, random_state=s)
        lab_s = gmm.fit_predict(X)
        results['time_gmm'][i] = time.time() - t0
        results['ari_gmm'][i] = adjusted_rand_score(y_true, lab_s)
        results['nmi_gmm'][i] = normalized_mutual_info_score(y_true, lab_s)

        print(f"{i+1}/{n_rep}  "
              f"GMM-grad: ARI {results['ari_grad'][:i+1].mean():.3f} NMI {results['nmi_grad'][:i+1].mean():.3f} | "
              f"GMM-GS: ARI {results['ari_gs'][:i+1].mean():.3f} NMI {results['nmi_gs'][:i+1].mean():.3f} | "
              f"GMM: ARI {results['ari_gmm'][:i+1].mean():.3f} NMI {results['nmi_gmm'][:i+1].mean():.3f}")

    print("Experiment loop finished.")
    return results