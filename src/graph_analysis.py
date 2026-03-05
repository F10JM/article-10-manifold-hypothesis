import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(points, k=6, metric='euclidean'):
    """
    Build a k-nn graph. Edges weighted by distance.
    For spherical embeddings, pass metric='cosine' and distances
    are converted to arc distance.
    Returns: NetworkX Graph
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    G = nx.Graph()
    G.add_nodes_from(range(len(points)))

    for i in range(len(points)):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            if metric == 'cosine':
                # Convert cosine distance to arc distance
                # cosine distance = 1 - cos(angle), so cos(angle) = 1 - d
                d = np.arccos(np.clip(1.0 - d, -1.0, 1.0))
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=d)

    return G


def compute_shortest_paths(graph, pairs):
    """
    Compute shortest path lengths for a list of (i, j) pairs.
    Returns array of distances. np.inf if disconnected.
    """
    distances = []
    for i, j in pairs:
        try:
            d = nx.shortest_path_length(graph, source=i, target=j, weight='weight')
        except nx.NetworkXNoPath:
            d = np.inf
        distances.append(d)
    return np.array(distances)


def sample_pairs(n, n_pairs=300, seed=42):
    """Random sample of index pairs for path comparison."""
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return list(pairs)
