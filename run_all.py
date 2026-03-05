"""
Reproduce the torus example from Section 3.5 of
"Statistical exploration of the Manifold Hypothesis" (Whiteley et al., 2025)

Usage: python run_all.py
Output: figures saved to figures/
"""

import os
import numpy as np
from src.torus import sample_torus
from src.lms_model import generate_lms_data
from src.pca_embedding import pca_embed, spherical_project
from src.graph_analysis import build_knn_graph, compute_shortest_paths, sample_pairs
from src.plotting import plot_torus, plot_pca_dims, plot_norms_kde, plot_isometry

import matplotlib
matplotlib.use('Agg')


def main():
    os.makedirs("figures", exist_ok=True)

    print("Step 1: Sampling torus (n=1500)...")
    Z, theta1, theta2 = sample_torus(n=1500)

    print("Step 2: Generating LMS data (p=2000, sigma=0.5)...")
    Y, X = generate_lms_data(Z, p=2000, sigma=0.5)

    print("Step 3: Computing PCA embedding (r=9 for viz, r=15 for isometry)...")
    zeta_9 = pca_embed(Y, r=9)
    zeta_15 = pca_embed(Y, r=15)

    print("Step 4: Spherical projection...")
    zeta_sp, norms = spherical_project(zeta_15)

    print("Step 5: Building k-nn graphs and computing shortest paths...")
    pairs = sample_pairs(len(Z), n_pairs=300)

    graph_Z = build_knn_graph(Z, k=8, metric='euclidean')
    graph_M = build_knn_graph(zeta_sp, k=8, metric='cosine')

    dist_Z = compute_shortest_paths(graph_Z, pairs)
    dist_M = compute_shortest_paths(graph_M, pairs)

    scaling_factor = np.sqrt(2)

    print("Step 6: Generating figures...")

    fig = plot_torus(Z, theta1, theta2)
    fig.savefig("figures/torus_latent_space.png", dpi=150)
    print("  -> figures/torus_latent_space.png")

    fig = plot_pca_dims(zeta_9, theta1, theta2)
    fig.savefig("figures/pca_embedding_dims.png", dpi=150)
    print("  -> figures/pca_embedding_dims.png")

    fig = plot_norms_kde(norms)
    fig.savefig("figures/embedding_norms_kde.png", dpi=150)
    print("  -> figures/embedding_norms_kde.png")

    fig = plot_isometry(dist_Z, dist_M, scaling_factor)
    fig.savefig("figures/isometry_verification.png", dpi=150)
    print("  -> figures/isometry_verification.png")

    print("Done. All figures saved to figures/")


if __name__ == "__main__":
    main()
