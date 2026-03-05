import numpy as np


def pca_embed(Y, r):
    """
    Compute dimension-r PCA embedding.
    Returns zeta: (n, r) array of principal component scores,
    scaled by p^{-1/2} as per Theorem 1.
    """
    n, p = Y.shape
    Y_centered = Y - Y.mean(axis=0)
    U, S, _ = np.linalg.svd(Y_centered, full_matrices=False)
    zeta = U[:, :r] * S[:r]
    zeta /= np.sqrt(p)
    return zeta


def spherical_project(zeta):
    """
    Project each row onto the unit hypersphere: zeta_sp_i = zeta_i / ||zeta_i||
    Returns zeta_sp: (n, r), norms: (n,)
    """
    norms = np.linalg.norm(zeta, axis=1)
    zeta_sp = zeta / norms[:, np.newaxis]
    return zeta_sp, norms
