# Article 10 — Statistical Exploration of the Manifold Hypothesis

**Paper:** *Statistical exploration of the Manifold Hypothesis*, Nick Whiteley, Annie Gray, Patrick Rubin-Delanchy (2025). Read before the Royal Statistical Society, October 2025 (JRSS-B discussion paper).

**Course:** Dimension Reduction & Manifold Learning, Universite Paris Dauphine - PSL

**Students:** Fadi Jemmali, Yessin Moakher, Belgacem Ben Ziada

---

## Overview

This project reproduces the **torus example from Section 3.5** of the paper. The authors propose the **Latent Metric Space (LMS) model** — a simple statistical model where high-dimensional observations arise from latent variables through correlated Gaussian processes plus noise:

$$Y_{ij} = X_j(z_i) + \sigma\,\varepsilon_{ij}, \qquad j = 1, \dots, p$$

where each feature $X_j$ is a Gaussian process indexed by latent variables $z_i$ living on a metric space $Z$, and $\varepsilon_{ij} \sim \mathcal{N}(0,1)$ is independent noise. The kernel function $f(z, z') = \exp(-\|z - z'\|^2)$ governs the covariance structure.

We illustrate three key theoretical results using a torus as the latent space:

| Result | Statement | What we show |
|--------|-----------|-------------|
| **Proposition 1** | Data inner products reflect manifold inner products | PCA embedding preserves torus topology |
| **Proposition 2** | The manifold $M$ is homeomorphic to $Z$ | Coloring by latent angles is preserved in PCA dimensions |
| **Proposition 3** | Under stationarity, $M$ is isometric to $Z$ up to scaling $\sqrt{-2g'(0)}$ | Graph geodesics on $\hat{M}$ vs $Z$ follow a line with slope $\sqrt{2}$ |
| **Theorem 1** | PCA embedding converges to the feature map as $p \to \infty$ | Increasing $p$ produces cleaner manifold recovery |

---

## Project Structure

```
article-10-manifold-hypothesis/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── torus.py              # Torus sampling utilities
│   ├── lms_model.py          # LMS data generation (GP kernel + Cholesky sampling)
│   ├── pca_embedding.py      # PCA embedding + spherical projection
│   ├── graph_analysis.py     # k-NN graph, shortest paths, isometry check
│   └── plotting.py           # All figure-generating functions
├── run_all.py                # Main script: generates all figures
├── run_experiments.py        # Varying-p experiment (Theorem 1)
├── notebooks/
│   └── Article 10 - code.ipynb   # Interactive notebook with inline figures
└── figures/                  # Output directory (created automatically)
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the Jupyter notebook, also install:
```bash
pip install ipykernel
python -m ipykernel install --user --name dimension-reduction --display-name "Dimension Reduction"
```

---

## Usage

### Generate all figures

```bash
python run_all.py
```

### Run the varying-p experiment

```bash
python run_experiments.py
```

### Interactive notebook

Open `notebooks/Article 10 - code.ipynb` in Jupyter or VSCode and select the **Dimension Reduction** kernel.

---

## Output Figures

| File | Description | Paper reference |
|------|-------------|-----------------|
| `figures/torus_latent_space.png` | 3D torus colored by $\theta_1$ and $\theta_2$ | Figure 3 |
| `figures/pca_embedding_dims.png` | PCA dimensions 1-3, 4-6, 7-9 colored by both angles | Figure 4 |
| `figures/embedding_norms_kde.png` | Distribution of PCA embedding norms $\|\zeta_i\|$ | Section 4.3 |
| `figures/isometry_verification.png` | Graph geodesics on $\hat{M}$ vs $Z$ with $\sqrt{2}$ reference line | Figure 5 |
| `figures/varying_p.png` | PCA dims 1-3 for $p \in \{200, 1000, 5000\}$ | Theorem 1 |

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $n$ | 1500 | Number of points sampled on the torus |
| $p$ | 2000 | Number of observed features |
| $\sigma$ | 0.5 | Noise standard deviation |
| $R$ | 2.0 | Torus major radius |
| $r$ | 1.0 | Torus minor radius |
| $k$ | 8 | Neighbors for k-NN graph |
| PCA dim (visualization) | 9 | Dimensions 1-9 for the 3x2 grid |
| PCA dim (isometry) | 15 | Higher dimension for geodesic computation |

---

## Method Summary

1. **Sample latent space** — $n$ points uniformly on a torus $\mathbb{T}^2 \subset \mathbb{R}^3$
2. **Generate LMS data** — Gaussian kernel matrix $K_{il} = \exp(-\|z_i - z_l\|^2)$, Cholesky factor $L$, signal $X = L \cdot \mathcal{N}(0, I_{n \times p})$, observation $Y = X + \sigma E$
3. **PCA embedding** — Top-$r$ principal component scores scaled by $p^{-1/2}$
4. **Spherical projection** — Normalize each embedding vector: $\tilde{\zeta}_i = \zeta_i / \|\zeta_i\|$
5. **Isometry check** — Build k-NN graphs on both $Z$ (Euclidean distances) and $\hat{M}$ (arc distances), compare shortest path lengths for random point pairs
