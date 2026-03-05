# CLAUDE_CODE_PLAN.md — Article 10 Code Demo

## Context

University exam project (Dimension Reduction & Manifold Learning, Dauphine PSL). We illustrate the **torus example from Section 3.5** of:

> **"Statistical exploration of the Manifold Hypothesis"**  
> Nick Whiteley, Annie Gray, Patrick Rubin-Delanchy (2025)  
> Read before the Royal Statistical Society, Oct 2025

The paper proposes the **Latent Metric Space (LMS) model**: a simple statistical model that produces manifold structure in high-dimensional data from latent variables, correlation, and stationarity. The torus example illustrates three key results:
1. Data inner products reflect manifold inner products (Proposition 1)
2. The manifold M is homeomorphic to the latent space Z (Proposition 2)
3. Under stationarity, M is isometric to Z up to scaling (Proposition 3)

---

## Project Structure

```
article-10-manifold-hypothesis/
├── CLAUDE_CODE_PLAN.md
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── torus.py              # torus sampling utilities
│   ├── lms_model.py          # LMS data generation (GP sampling + noise)
│   ├── pca_embedding.py      # PCA embedding + spherical projection
│   ├── graph_analysis.py     # k-nn graph, shortest paths, isometry check
│   └── plotting.py           # all figure-generating functions
├── run_all.py                # main script: generates everything, saves figures
├── run_experiments.py        # bonus: varying p experiment
└── figures/                  # output directory (created automatically)
```

**No notebooks.** Everything is `.py` files. `run_all.py` is the single entry point that generates all figures into `figures/`. Later we can convert the results into a notebook for submission if needed.

---

## Step-by-step Plan

### Step 1 — `src/torus.py`

Sample latent variables on a torus.

**Paper ref:** Section 3.5, Figure 3

```python
def sample_torus(n, R=2.0, r=1.0, seed=42):
    """
    Sample n points uniformly on a torus in R^3.
    Returns:
        Z: (n, 3) array of 3D coordinates
        theta1: (n,) angles around the big circle [0, 2π)
        theta2: (n,) angles around the tube [0, 2π)
    """
```

- Major radius R=2, minor radius r=1
- θ₁, θ₂ ~ Uniform[0, 2π)
- x = (R + r·cos(θ₂))·cos(θ₁), y = (R + r·cos(θ₂))·sin(θ₁), z = r·sin(θ₂)

### Step 2 — `src/lms_model.py`

Generate high-dimensional data from the LMS model.

**Paper ref:** Section 2 (Eq. 1), Section 3.5

```python
def gaussian_kernel_matrix(Z, lengthscale=1.0):
    """Compute K where K_il = exp(-||Z_i - Z_l||^2 / lengthscale^2)"""

def generate_lms_data(Z, p=500, sigma=0.5, seed=42):
    """
    Generate Y under the LMS model: Y_ij = X_j(Z_i) + sigma * E_ij

    - Compute kernel matrix K from Z
    - Cholesky: K + jitter = L @ L^T
    - Signal: X = L @ random_normal(n, p)  (all p columns at once)
    - Noise: E = random_normal(n, p)
    - Y = X + sigma * E

    Returns: Y (n, p), X (n, p)
    """
```

**Key:** Sample all p GP columns at once via matrix multiply, don't loop.

### Step 3 — `src/pca_embedding.py`

PCA embedding and spherical projection.

**Paper ref:** Section 4.1 (Theorem 1), Section 4.3

```python
def pca_embed(Y, r):
    """
    Compute dimension-r PCA embedding.
    Returns zeta: (n, r) array of principal component scores,
    scaled by p^{-1/2} as per Theorem 1.
    """

def spherical_project(zeta):
    """
    Project each row onto the unit hypersphere: zeta_sp_i = zeta_i / ||zeta_i||
    Returns zeta_sp: (n, r), norms: (n,)
    """
```

### Step 4 — `src/graph_analysis.py`

Nearest neighbor graphs and shortest paths for isometry verification.

**Paper ref:** Section 4.4, Proposition 3, Figure 5

```python
def build_knn_graph(points, k=6, metric='euclidean'):
    """
    Build a k-nn graph. Edges weighted by distance.
    For spherical embeddings, pass metric='cosine' and convert to arc distance.
    Returns: NetworkX Graph
    """

def compute_shortest_paths(graph, pairs):
    """
    Compute shortest path lengths for a list of (i, j) pairs.
    Returns array of distances. np.inf if disconnected.
    """

def sample_pairs(n, n_pairs=300, seed=42):
    """Random sample of index pairs for path comparison."""
```

**Arc distance:** d_S(x, y) = arccos(clip(x·y, -1, 1)). Use this as edge weight for the spherical embedding graph.

**Geodesic on torus:** Use graph-based shortest paths on the 3D torus points (same method as for M). Do NOT use a closed-form formula — the paper uses graph geodesics for both.

### Step 5 — `src/plotting.py`

All visualization functions. Each returns a matplotlib Figure so `run_all.py` can save them.

```python
def plot_torus(Z, theta1, theta2):
    """
    Figure 1: two 3D scatter plots of torus, colored by θ₁ and θ₂.
    Reproduces Figure 3 of the paper.
    """

def plot_pca_dims(zeta, theta1, theta2):
    """
    Figure 2: 3x2 grid of 3D scatters.
    Top row: dims (1-3), (4-6), (7-9) colored by θ₁
    Bottom row: same colored by θ₂
    Reproduces Figure 4 of the paper.
    """

def plot_norms_kde(norms):
    """
    Figure 3: KDE of PCA embedding norms ||ζ_i||.
    Shows the variation that spherical projection removes.
    """

def plot_isometry(dist_Z, dist_M, scaling_factor):
    """
    Figure 4: scatter of shortest path lengths in M vs Z.
    Overlay red line with slope = scaling_factor (√2).
    Reproduces Figure 5 of the paper.
    """

def plot_varying_p(results_dict):
    """
    Figure 5: 1x3 grid showing PCA dims 1-3 for p=100, 500, 2000.
    Shows manifold recovery improving with p (Theorem 1).
    """
```

**Style rules:**
- White background, fontsize 12, `tight_layout()`
- **Cyclic colormap** (`twilight_shifted`) for angles — prevents false discontinuity at 0/2π
- Consistent figure sizes: single plots (8, 6), grids (16, 10)
- Save at 150 dpi for reasonable file size
- Small scatter point size (s=3 or s=5) since n is large

### Step 6 — `run_all.py`

Main entry point. Runs everything, prints progress, saves all figures.

```python
"""
Reproduce the torus example from Section 3.5 of
"Statistical exploration of the Manifold Hypothesis" (Whiteley et al., 2025)

Usage: python run_all.py
Output: figures saved to figures/
"""

def main():
    os.makedirs("figures", exist_ok=True)

    print("Step 1: Sampling torus...")
    # sample_torus(n=2000)

    print("Step 2: Generating LMS data (p=500, sigma=0.5)...")
    # generate_lms_data(Z, p=500, sigma=0.5)

    print("Step 3: Computing PCA embedding...")
    # pca_embed(Y, r=9) for visualization
    # pca_embed(Y, r=15) for isometry verification

    print("Step 4: Spherical projection...")
    # spherical_project(zeta)

    print("Step 5: Building k-nn graphs and computing shortest paths...")
    # build graphs on Z (Euclidean) and zeta_sp (arc distance)
    # compute_shortest_paths for 300 sampled pairs

    print("Step 6: Generating figures...")
    # plot_torus → figures/torus_latent_space.png
    # plot_pca_dims → figures/pca_embedding_dims.png
    # plot_norms_kde → figures/embedding_norms_kde.png
    # plot_isometry → figures/isometry_verification.png

    print("Done. Figures saved to figures/")
```

### Step 7 — `run_experiments.py`

Bonus experiment: effect of p on manifold recovery.

```python
"""
Show how increasing p improves PCA recovery of the manifold (Theorem 1).
Fixes n=1000, sigma=0.5, varies p in {100, 500, 2000}.

Usage: python run_experiments.py
Output: figures/varying_p.png
"""
```

---

## Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| n | 2000 | Enough points, Cholesky still fast (~32MB kernel matrix) |
| p | 500 | 500 gives good signal with manageable compute |
| σ | 0.5 | Moderate noise, manifold still visible |
| R (major radius) | 2.0 | Standard torus |
| r (minor radius) | 1.0 | Standard torus |
| k (k-nn graph) | 6 | Reasonable for 2D manifold |
| PCA dim (visualization) | 9 | To show dims 1-3, 4-6, 7-9 like Figure 4 |
| PCA dim (isometry) | 15 | Need enough dims for geodesic computation |
| n_pairs (isometry plot) | 300 | Enough for a clear scatter, fast to compute |

If n=2000 is slow (Cholesky > 30s), drop to n=1000.

---

## Technical Notes

- **GP sampling:** `X = L @ np.random.randn(n, p)` where `L = cholesky(K + 1e-5*I)`. One matrix multiply, no loops.
- **Arc distance:** `d = np.arccos(np.clip(np.dot(x, y), -1, 1))`. Always clip before arccos.
- **k-nn construction:** `sklearn.neighbors.NearestNeighbors` → extract distances and indices → build `networkx.Graph` with distance weights.
- **Shortest paths:** `networkx.shortest_path_length` with weight='weight'. Only compute for sampled pairs, not all-pairs.
- **Isometry scaling factor:** For f(z,z') = exp(-||z-z'||²), g(t) = exp(-t), g'(0) = -1, scaling = √(-2g'(0)) = √2.
- **Colormap:** `twilight_shifted` for angles (cyclic). Consistent across all angle-colored plots.

---

## Expected Output

After `python run_all.py`:

| File | Description |
|------|-------------|
| `figures/torus_latent_space.png` | 3D torus colored by θ₁ and θ₂ (≈ Figure 3) |
| `figures/pca_embedding_dims.png` | 3×2 grid of PCA dims 1-9 colored by both angles (≈ Figure 4) |
| `figures/embedding_norms_kde.png` | KDE of ‖ζ_i‖ showing amplitude variation |
| `figures/isometry_verification.png` | Geodesic distances M vs Z with √2 line (≈ Figure 5) |

After `python run_experiments.py`:

| File | Description |
|------|-------------|
| `figures/varying_p.png` | Effect of increasing p on manifold recovery |

---

## requirements.txt

```
numpy
scipy
scikit-learn
networkx
matplotlib
```

---

## README.md should contain

- Paper title, authors, journal (JRSS-B discussion paper, 2025)
- Article number: 10
- Student names: [FILL IN]
- One paragraph: what the code demonstrates
- How to install: `pip install -r requirements.txt`
- How to run: `python run_all.py` and `python run_experiments.py`
- Table of output figures with one-line descriptions
- Brief explanation of the LMS model and the torus example

---

## After Completion Checklist

- [ ] `pip install -r requirements.txt` works
- [ ] `python run_all.py` runs without errors in < 5 minutes
- [ ] `python run_experiments.py` runs without errors
- [ ] All figures saved to `figures/` and look clean
- [ ] Isometry plot shows clear linear relationship with slope ≈ √2
- [ ] PCA dims plot shows torus topology preserved in coloring
- [ ] README is filled in
- [ ] No hardcoded absolute paths
