import numpy as np

def create_cov_matrix(theta, rho, m):
  sigma = np.zeros((m, m))
  for i in range(m):
    for j in range(m):
      sigma[i,j] = theta*rho**abs(i-j)
  return sigma

def create_mu_vector(m, rng):
  mu = rng.normal(0, 1, size=m)
  mu = (mu / np.linalg.norm(mu))*2
  return mu

# def generate_world_parameters(m_features, k_components, world_seed):
#     """
#     Generates the true parameters (means and covariances) for a single 'world'.
#     """
#     rng = np.random.default_rng(world_seed)
    
#     true_mus = []
#     true_covs = []
    
#     for _ in range(k_components):
#         theta = 10 ** rng.uniform(-1, 2)
#         rho = rng.uniform(-0.9, 0.9)
        
#         cov = create_cov_matrix(theta, rho, m_features)
#         mu = create_mu_vector(m_features, rng)
        
#         true_mus.append(mu)
#         true_covs.append(cov)
        
#     return true_mus, true_covs
def generate_world_parameters(m_features, k_components, world_seed):
    """
     Generates the true parameters (means and covariances) for a single 'world'.
     """
    np.random.seed(world_seed)

    def sample_theta_rho():
        theta = 10 ** np.random.uniform(-1, 2)
        rho = np.random.uniform(-0.9, 0.9)
        return theta, rho

    def create_mu_vector(m):
        mu = np.random.normal(0, 1, size=m)
        mu = (mu / np.linalg.norm(mu))*2
        return mu

    true_mus = []
    true_covs = []

    thetas_rhos = [sample_theta_rho() for _ in range(k_components)]
    for theta, rho in thetas_rhos:
        cov = create_cov_matrix(theta, rho, m_features)
        true_covs.append(cov)

    for _ in range(k_components):
        mu = create_mu_vector(m_features)
        true_mus.append(mu)

    return true_mus, true_covs


def generate_trial_dataset(n_samples_per_component, true_mus, true_covs, trial_seed):
    np.random.seed(trial_seed)
    k_components = len(true_mus)
    
    X_components = []
    for k in range(k_components):
        X_k = np.random.multivariate_normal(true_mus[k], true_covs[k], size=n_samples_per_component)
        X_components.append(X_k)
        
    X_full = np.vstack(X_components)
    y_true = np.repeat(np.arange(k_components), n_samples_per_component)
    
    permutation = np.random.permutation(X_full.shape[0])
    X_shuffled = X_full[permutation]
    y_shuffled = y_true[permutation]
    
    return X_shuffled, y_shuffled