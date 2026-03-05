"""
Show how increasing p improves PCA recovery of the manifold (Theorem 1).
Fixes n=1000, sigma=0.5, varies p in {100, 500, 2000}.

Usage: python run_experiments.py
Output: figures/varying_p.png
"""

import os
import numpy as np
from src.torus import sample_torus
from src.lms_model import generate_lms_data
from src.pca_embedding import pca_embed
from src.plotting import plot_varying_p

import matplotlib
matplotlib.use('Agg')


def main():
    os.makedirs("figures", exist_ok=True)

    print("Sampling torus (n=1000)...")
    Z, theta1, theta2 = sample_torus(n=1000, seed=42)

    p_values = [200, 1000, 5000]
    results = {}

    for p in p_values:
        print(f"Generating LMS data with p={p}...")
        Y, _ = generate_lms_data(Z, p=p, sigma=0.5, seed=42)
        zeta = pca_embed(Y, r=3)
        results[p] = zeta

    print("Generating figure...")
    fig = plot_varying_p(results, theta1)
    fig.savefig("figures/varying_p.png", dpi=150)
    print("  -> figures/varying_p.png")

    print("Done.")


if __name__ == "__main__":
    main()
