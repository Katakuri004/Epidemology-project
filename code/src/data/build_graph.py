from typing import Tuple, Optional

import numpy as np


def add_self_loops(adjacency: np.ndarray) -> np.ndarray:
    adj = adjacency.copy()
    np.fill_diagonal(adj, adj.diagonal() + 1.0)
    return adj


def normalize_symmetric(adjacency_with_loops: np.ndarray) -> np.ndarray:
    degrees = adjacency_with_loops.sum(axis=1)
    # Avoid division-by-zero
    degrees = np.where(degrees > 0.0, degrees, 1.0)
    inv_sqrt_deg = np.power(degrees, -0.5)
    d_mat = np.diag(inv_sqrt_deg)
    return d_mat @ adjacency_with_loops @ d_mat


def build_normalized_graph(adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (adjacency_with_self_loops, normalized_adjacency)
    """
    a_hat = add_self_loops(adjacency)
    norm = normalize_symmetric(a_hat)
    return a_hat, norm


# --- New helpers for real graphs ---
def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Pairwise haversine distance matrix in kilometers.
    Inputs in degrees, returns (N, N) matrix.
    """
    R = 6371.0
    lat1_r = np.radians(lat1)[:, None]
    lon1_r = np.radians(lon1)[:, None]
    lat2_r = np.radians(lat2)[None, :]
    lon2_r = np.radians(lon2)[None, :]
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def adjacency_from_distance_threshold(coords: np.ndarray, radius_km: float) -> np.ndarray:
    """Build symmetric adjacency where edge=1 if distance<=radius_km and i!=j."""
    lat = coords[:, 0]
    lon = coords[:, 1]
    D = _haversine_km(lat, lon, lat, lon)
    A = (D <= radius_km).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A


def adjacency_from_knn(coords: np.ndarray, k: int, seed: Optional[int] = None) -> np.ndarray:
    """Symmetric k-NN adjacency by distance."""
    lat = coords[:, 0]
    lon = coords[:, 1]
    D = _haversine_km(lat, lon, lat, lon)
    np.fill_diagonal(D, np.inf)
    # Deterministic tie-breaker via tiny jitter controlled by seed
    if seed is not None:
        rng = np.random.default_rng(seed)
        jitter = rng.normal(scale=1e-6, size=D.shape)
        jitter[np.eye(D.shape[0], dtype=bool)] = 0.0
        D = D + jitter
    N = coords.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    idx = np.argsort(D, axis=1)[:, :k]
    for i in range(N):
        A[i, idx[i]] = 1.0
    # symmetrize
    A = np.maximum(A, A.T)
    return A


def build_norm_from_coords(coords: np.ndarray, mode: str = "distance_threshold", radius_km: float = 1000.0, k: int = 5, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "distance_threshold":
        A = adjacency_from_distance_threshold(coords, radius_km)
    elif mode == "knn":
        A = adjacency_from_knn(coords, k, seed=seed)
    else:
        raise ValueError("mode must be 'distance_threshold' or 'knn'")
    return build_normalized_graph(A)

