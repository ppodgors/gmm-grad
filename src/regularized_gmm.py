import numpy as np
import torch
import torch.optim as optim
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal

class RegularizedGMM:
    def __init__(self, k_components, max_iters=200, tol=1e-9, eta_method='grad'):
        if eta_method not in ['grad', 'gs']:
            raise ValueError("eta_method must be either 'grad' or 'gs'")
        self.k_components = k_components
        self.max_iters = max_iters
        self.tol = tol
        self.eta_method = eta_method
        self.pi_, self.means_, self.covariances_, self.seed_ = None, None, None, None

    def _initialize_params(self, X):
        gmm = GaussianMixture(n_components=self.k_components, max_iter=5, random_state=self.seed_, init_params='kmeans', n_init=10)
        gmm.fit(X)
        return gmm.weights_.copy(), gmm.means_.copy(), gmm.covariances_.copy()

    def _e_step(self, X, pi, means, covariances):
        log_p = np.zeros((X.shape[0], self.k_components))
        for k in range(self.k_components):
            stable_cov = covariances[k] + 1e-6 * np.eye(covariances[k].shape[0])
            log_p[:, k] = np.log(pi[k] + 1e-12) + multivariate_normal.logpdf(X, means[k], stable_cov, allow_singular=True)
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def _m_step(self, X, p, T, reg_eta):
        n, m = X.shape
        pi_new = p.sum(axis=0) / n
        w_new = p / (p.sum(axis=0) + 1e-12)
        means_new = np.dot(w_new.T, X)
        covariances_new = []
        for k in range(self.k_components):
            if np.sum(p[:,k]) < 1e-6:
                cov_reg = np.eye(X.shape[1]) * 1e-6
            else:
                diff = X - means_new[k]
                w_k = w_new[:, k]
                cov_k = (w_k[:, None] * diff).T @ diff / np.sum(w_k)
                beta_k = (n * pi_new[k]) / (reg_eta[k] + n * pi_new[k])
                cov_reg = beta_k * cov_k + (1 - beta_k) * T[k]
                cov_reg += np.eye(cov_reg.shape[0]) * 1e-6
                cov_reg = (cov_reg + cov_reg.T) / 2
            covariances_new.append(cov_reg)
        return pi_new, means_new, np.array(covariances_new)

    def _compute_log_likelihood(self, X, pi, means, covariances):
        log_sum = None
        for k in range(self.k_components):
            Sigma = np.nan_to_num(covariances[k], nan=1e-6) + np.eye(covariances[k].shape[0]) * 1e-6
            log_lik_k = np.log(pi[k] + 1e-12) + multivariate_normal.logpdf(X, means[k], Sigma, allow_singular=True)
            if k == 0:
                log_sum = log_lik_k
            else:
                log_sum = np.logaddexp(log_sum, log_lik_k)
        return log_sum.sum()
    
    def _assign_labels(self, X, pi, means, covariances):
        log_p = np.zeros((X.shape[0], self.k_components))
        for k in range(self.k_components):
            try:
                current_cov = covariances[k]
                if not np.all(np.isfinite(current_cov)):
                    raise ValueError("Covariance matrix contains NaN or Inf")
                log_p[:, k] = np.log(pi[k] + 1e-12) + multivariate_normal.logpdf(X, means[k], current_cov, allow_singular=True)
            except (np.linalg.LinAlgError, ValueError):
                log_p[:, k] = -np.inf
        return np.argmax(log_p, axis=1)

    def _calculate_eta(self, X, pi, means, covariances, T,adam):
        if self.eta_method == 'grad':
            return self._eta_grad(X, pi, means, covariances, T, adam)
        else:
            return self._eta_gs(X, pi, means, covariances, T)

    def _eta_gs(self, X, pi, means, covariances, T):
        L=5; n, m = X.shape; labels = self._assign_labels(X, pi, means, covariances)
        clusters = {k: X[labels == k] for k in range(len(pi))}
        final_eta = []; eta_list = [0] + list(np.logspace(0, 4, num=5))
        for k in range(len(pi)):
            if k not in clusters or clusters[k].shape[0] < 4:
                final_eta.append(1); continue
            cluster = clusters[k]
            kf = KFold(n_splits=min(L, cluster.shape[0] - 1)); Tk = T[k]
            errors = np.zeros(len(eta_list))
            for train_index, test_index in kf.split(cluster):
                train_data, test_data = cluster[train_index], cluster[test_index]
                if train_data.shape[0] < 2: continue
                n_train = train_data.shape[0]
                S_train = np.cov(train_data, rowvar=False, bias=True)
                S_val = np.eye(m) if test_data.shape[0] < 2 else np.cov(test_data, rowvar=False, bias=True)
                S_val, S_train, Tk = np.atleast_2d(S_val), np.atleast_2d(S_train), np.atleast_2d(Tk)
                for j, eta in enumerate(eta_list):
                    Sigma_eta = n_train / (eta + n_train) * S_train + (eta) / (eta + n_train) * Tk
                    Sigma_eta = np.atleast_2d(Sigma_eta) + np.eye(Sigma_eta.shape[0]) * 1e-6
                    try:
                        trace_term = np.trace(np.linalg.solve(Sigma_eta, S_val))
                        log_det_term = np.log(np.linalg.det(Sigma_eta) + 1e-12)
                        errors[j] += trace_term + log_det_term
                    except np.linalg.LinAlgError: errors[j] += np.inf
            final_eta.append(eta_list[np.argmin(errors)])
        return final_eta
    
    def _eta_grad(self, X, pi, means, covariances, T, use_adam=False):
        labels = self._assign_labels(X, pi, means, covariances)
        clusters = {k: torch.tensor(X[labels == k], dtype=torch.float32) for k in range(len(pi))}
        T_torch = [torch.tensor(t, dtype=torch.float32) for t in T]
        final_eta = []

        for k in range(len(pi)):
            if k not in clusters or clusters[k].shape[0] < 4:
                final_eta.append((clusters.get(k, np.empty((0,0))).shape[0] + 1)**2)
                continue
            
            cluster = clusters[k]
            n_k = cluster.shape[0]
            L_eff = min(3 if n_k < 50 else 5, n_k // 2)

            if L_eff < 2:
                final_eta.append((n_k + 1)**2)
                continue

            kf = KFold(n_splits=L_eff, shuffle=True)
            cov_pairs = []

            for tr, va in kf.split(cluster):
                Xtr, Xva = cluster[tr], cluster[va]
                if Xtr.shape[0] < 2 or Xva.shape[0] < 2:
                    continue
                n_tr = Xtr.shape[0]
                S_tr = (Xtr - Xtr.mean(0)).T @ (Xtr - Xtr.mean(0)) / n_tr
                S_val = (Xva - Xva.mean(0)).T @ (Xva - Xva.mean(0)) / Xva.shape[0]
                cov_pairs.append((S_tr, S_val, n_tr))

            if not cov_pairs:
                final_eta.append((n_k + 1)**2)
                continue

            eta0_val = 1
            if use_adam:
                results = {}
                eta_min = 1
                eta_max = 100 * n_k
                for tag, eta0_val in (('small', eta_min), ('large', eta_max)):
                    eta = torch.tensor([eta0_val], dtype=torch.float32, requires_grad=True)
                    optimizer = optim.Adam([eta], lr=0.02)
                    no_improve, prev_err = 0, float('inf')
                    for epoch in range(20):
                        total_err = 0.0
                        for S_tr, S_val, n_tr in cov_pairs:
                            total_err += self._compute_pytorch_error(eta.clamp(min=1e-6), S_tr, S_val, T_torch[k], n_tr)
                        
                        optimizer.zero_grad()
                        total_err.backward()
                        optimizer.step()
                        
                        if eta.grad is not None and eta.grad.abs().item() < 1e-5:
                            break
                        
                        cur_err = total_err.item()
                        rel_err = abs(prev_err - cur_err) / (abs(prev_err) + 1e-8)
                        
                        if rel_err < 1e-6:
                            no_improve += 1
                            if no_improve >= 10:
                                break
                        else:
                            no_improve = 0
                        prev_err = cur_err
                    results[tag] = (eta.clamp(min=1e-6).item(), total_err.item())
                
                best_tag = min(results, key=lambda t: results[t][1])
                best_eta = results[best_tag][0]
                final_eta.append(best_eta)
            else:
                log_eta = torch.tensor([np.log(eta0_val)], dtype=torch.float32, requires_grad=True)
                optimizer = optim.LBFGS([log_eta], history_size=5, max_iter=30,
                                        line_search_fn="strong_wolfe", tolerance_grad=1e-5, tolerance_change=1e-4)
                
                def closure():
                    optimizer.zero_grad()
                    eta_from_log = torch.exp(log_eta) + 1e-7
                    total_err = 0.0
                    for S_tr, S_val, n_tr in cov_pairs:
                        total_err += self._compute_pytorch_error(eta_from_log, S_tr, S_val, T_torch[k], n_tr)
                    total_err.backward()
                    return total_err
                    
                optimizer.step(closure)
                best_eta = (torch.exp(log_eta) + 1e-7).item()
                final_eta.append(best_eta)
                
        return final_eta
    
    def _compute_pytorch_error(self, eta, S_tr, S_val, Tk, n_tr: int):
        dtype, device = S_tr.dtype, S_tr.device; eta = eta.to(dtype=dtype, device=device)
        S_val, Tk = S_val.to(dtype=dtype, device=device), Tk.to(dtype=dtype, device=device)
        beta = n_tr / (eta + n_tr); S_eta = beta * S_tr + (1.0 - beta) * Tk
        try: nudge = (torch.mean(torch.abs(torch.diag(S_eta))) * 1e-6 + 1e-9) if S_eta.numel() > 0 else 1e-9
        except: nudge = 1e-9
        eye = torch.eye(S_eta.shape[0], dtype=dtype, device=device); S_eta_stable = S_eta + eye * nudge
        try:
            trace = torch.trace(torch.linalg.solve(S_eta_stable, S_val)); logdet = torch.logdet(S_eta_stable)
            if torch.isnan(trace) or torch.isinf(trace) or torch.isnan(logdet) or torch.isinf(logdet):
                return eta.square().squeeze() * 1e9
            return trace + logdet
        except torch.linalg.LinAlgError: return eta.square().squeeze() * 1e9

    def fit(self, X, seed=42):
        self.seed_ = seed
        pi, means, covariances = self._initialize_params(X)
        T = []
        for cov in covariances:
            m = cov.shape[0]
            Ti = np.eye(m) * (np.trace(cov) / m) if m > 0 else np.eye(m)
            T.append(Ti)
        T = np.array(T)
        prev_log_likelihood = -np.inf
        for i in range(self.max_iters):
            if i == 0 or (i % 20 == 0 and i > 1):
                reg_eta = self._calculate_eta(X, pi, means, covariances, T, adam=(i==0))
            p = self._e_step(X, pi, means, covariances)
            pi, means, covariances = self._m_step(X, p, T, reg_eta)
            log_likelihood = self._compute_log_likelihood(X, pi, means, covariances)
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
        self.pi_ = pi; self.means_ = means; self.covariances_ = covariances
        return self

    def predict(self, X):
        if self.pi_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._assign_labels(X, self.pi_, self.means_, self.covariances_)