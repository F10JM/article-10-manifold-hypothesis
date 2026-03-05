import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


CMAP = 'twilight_shifted'


def plot_torus(Z, theta1, theta2):
    """
    Figure 1: two 3D scatter plots of torus, colored by theta1 and theta2.
    """
    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=theta1, cmap=CMAP, s=2)
    ax1.set_title(r'Torus colored by $\theta_1$', fontsize=12)
    fig.colorbar(sc1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=theta2, cmap=CMAP, s=2)
    ax2.set_title(r'Torus colored by $\theta_2$', fontsize=12)
    fig.colorbar(sc2, ax=ax2, shrink=0.6)

    fig.tight_layout()
    return fig


def plot_pca_dims(zeta, theta1, theta2):
    """
    Figure 2: 3x2 grid of 3D scatters.
    Top row: dims (1-3), (4-6), (7-9) colored by theta1
    Bottom row: same colored by theta2
    """
    fig = plt.figure(figsize=(16, 10))
    dim_groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]

    for col, (d1, d2, d3) in enumerate(dim_groups):
        # Top row: colored by theta1
        ax = fig.add_subplot(2, 3, col + 1, projection='3d')
        ax.scatter(zeta[:, d1], zeta[:, d2], zeta[:, d3], c=theta1, cmap=CMAP, s=2)
        ax.set_title(f'Dims {d1+1}-{d3+1}, ' + r'$\theta_1$', fontsize=12)

        # Bottom row: colored by theta2
        ax = fig.add_subplot(2, 3, col + 4, projection='3d')
        ax.scatter(zeta[:, d1], zeta[:, d2], zeta[:, d3], c=theta2, cmap=CMAP, s=2)
        ax.set_title(f'Dims {d1+1}-{d3+1}, ' + r'$\theta_2$', fontsize=12)

    fig.tight_layout()
    return fig


def plot_norms_kde(norms):
    """
    Figure 3: KDE of PCA embedding norms ||zeta_i||.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(norms, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(r'$\|\zeta_i\|$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of PCA embedding norms', fontsize=12)
    fig.tight_layout()
    return fig


def plot_isometry(dist_Z, dist_M, scaling_factor):
    """
    Figure 4: scatter of shortest path lengths in M vs Z.
    Overlay red line with slope = scaling_factor (sqrt(2)).
    """
    # Filter out inf
    mask = np.isfinite(dist_Z) & np.isfinite(dist_M)
    dz, dm = dist_Z[mask], dist_M[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dz, dm, s=5, alpha=0.5, label='Point pairs')

    max_val = max(dz.max(), dm.max() / scaling_factor) if len(dz) > 0 else 1
    line_x = np.linspace(0, max_val, 100)
    ax.plot(line_x, scaling_factor * line_x, 'r-', linewidth=2,
            label=rf'Slope = $\sqrt{{2}}$ = {scaling_factor:.3f}')

    ax.set_xlabel('Graph geodesic on Z (torus)', fontsize=12)
    ax.set_ylabel(r'Graph geodesic on $\hat{M}$ (spherical PCA)', fontsize=12)
    ax.set_title('Isometry verification (Proposition 3)', fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def plot_varying_p(results_dict, theta1):
    """
    Figure 5: 1x3 grid showing PCA dims 1-3 for varying p.
    """
    p_values = sorted(results_dict.keys())
    fig = plt.figure(figsize=(16, 5))

    for idx, p in enumerate(p_values):
        zeta = results_dict[p]
        ax = fig.add_subplot(1, len(p_values), idx + 1, projection='3d')
        ax.scatter(zeta[:, 0], zeta[:, 1], zeta[:, 2], c=theta1, cmap=CMAP, s=2)
        ax.set_title(f'p = {p}, dims 1-3', fontsize=12)

    fig.tight_layout()
    return fig
