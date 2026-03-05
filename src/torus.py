import numpy as np


def sample_torus(n, R=2.0, r=1.0, seed=42):
    """
    Sample n points uniformly on a torus in R^3.
    Returns:
        Z: (n, 3) array of 3D coordinates
        theta1: (n,) angles around the big circle [0, 2pi)
        theta2: (n,) angles around the tube [0, 2pi)
    """
    rng = np.random.default_rng(seed)
    theta1 = rng.uniform(0, 2 * np.pi, n)
    theta2 = rng.uniform(0, 2 * np.pi, n)

    x = (R + r * np.cos(theta2)) * np.cos(theta1)
    y = (R + r * np.cos(theta2)) * np.sin(theta1)
    z = r * np.sin(theta2)

    Z = np.column_stack([x, y, z])
    return Z, theta1, theta2
