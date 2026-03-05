import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky


def gaussian_kernel_matrix(Z, lengthscale=1.0):
    """Compute K where K_il = exp(-||Z_i - Z_l||^2 / lengthscale^2)"""
    sq_dists = cdist(Z, Z, metric='sqeuclidean')
    return np.exp(-sq_dists / lengthscale**2)


def generate_lms_data(Z, p=500, sigma=0.5, lengthscale=1.0, seed=42):
    """
    Generate Y under the LMS model: Y_ij = X_j(Z_i) + sigma * E_ij

    - Compute kernel matrix K from Z
    - Cholesky: K + jitter = L @ L^T
    - Signal: X = L @ random_normal(n, p)
    - Noise: E = random_normal(n, p)
    - Y = X + sigma * E

    Returns: Y (n, p), X (n, p)
    """
    rng = np.random.default_rng(seed)
    n = Z.shape[0]

    K = gaussian_kernel_matrix(Z, lengthscale=lengthscale)
    K += 1e-5 * np.eye(n)
    L = cholesky(K, lower=True)

    X = L @ rng.standard_normal((n, p))
    E = rng.standard_normal((n, p))
    Y = X + sigma * E

    return Y, X
