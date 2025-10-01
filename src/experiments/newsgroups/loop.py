import os
import pickle
import time
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
from src.regularized_gmm import RegularizedGMM

def execute_experiment_loop(data_dir, n_worlds, n_reps):
    results = {
        'GMM': {'ari': np.zeros((n_worlds, n_reps)), 'nmi': np.zeros((n_worlds, n_reps)), 'time': np.zeros((n_worlds, n_reps))},
        'GMM-GS': {'ari': np.zeros((n_worlds, n_reps)), 'nmi': np.zeros((n_worlds, n_reps)), 'time': np.zeros((n_worlds, n_reps))},
        'GMM-grad': {'ari': np.zeros((n_worlds, n_reps)), 'nmi': np.zeros((n_worlds, n_reps)), 'time': np.zeros((n_worlds, n_reps))}
    }

    for world_id in tqdm(range(n_worlds), desc="Processing Worlds"):
        data_file = os.path.join(data_dir, f'data_world_{world_id}.pkl')
        if not os.path.exists(data_file):
            tqdm.write(f"\nWarning: Data file not found: {data_file}. Skipping world {world_id}.")
            for method in results:
                results[method]['ari'][world_id, :] = np.nan
                results[method]['nmi'][world_id, :] = np.nan
                results[method]['time'][world_id, :] = np.nan
            continue
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            X, y_true = data['vectors'], data['labels']
            n_clusters = len(data['categories'])

        for rep_id in range(n_reps):
            rep_seed = world_id * 1000 + rep_id
            
            try:
                # GMM
                t0 = time.time()
                gmm = GaussianMixture(n_components=n_clusters, random_state=rep_seed)
                gmm_labels = gmm.fit_predict(X)
                results['GMM']['time'][world_id, rep_id] = time.time() - t0
                results['GMM']['ari'][world_id, rep_id] = adjusted_rand_score(y_true, gmm_labels)
                results['GMM']['nmi'][world_id, rep_id] = normalized_mutual_info_score(y_true, gmm_labels)

                # GMM-GS
                t0 = time.time()
                gmm_GS = RegularizedGMM(k_components=n_clusters, eta_method='gs')
                gmm_GS.fit(X, seed=rep_seed)
                GS_labels = gmm_GS.predict(X)
                results['GMM-GS']['time'][world_id, rep_id] = time.time() - t0
                results['GMM-GS']['ari'][world_id, rep_id] = adjusted_rand_score(y_true, GS_labels)
                results['GMM-GS']['nmi'][world_id, rep_id] = normalized_mutual_info_score(y_true, GS_labels)
                
                # GMM-grad
                t0 = time.time()
                gmm_grad = RegularizedGMM(k_components=n_clusters, eta_method='grad')
                gmm_grad.fit(X, seed=rep_seed)
                grad_labels = gmm_grad.predict(X)
                results['GMM-grad']['time'][world_id, rep_id] = time.time() - t0
                results['GMM-grad']['ari'][world_id, rep_id] = adjusted_rand_score(y_true, grad_labels)
                results['GMM-grad']['nmi'][world_id, rep_id] = normalized_mutual_info_score(y_true, grad_labels)

            except Exception as e:
                for method in results:
                    for metric in results[method]:
                        results[method][metric][world_id, rep_id] = np.nan
    return results