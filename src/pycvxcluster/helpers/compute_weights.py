from scipy.spatial import KDTree
from scipy.sparse import lil_array
from scipy.sparse import csr_array
from scipy.sparse import find
from scipy.sparse import triu
import scipy.linalg as la
import numpy as np
import numpy.typing as npt
from typing import Any
import time


def compute_weights(
    X: npt.ArrayLike, k: int, phi: float, gamma: float, verbose=1
) -> (npt.ArrayLike, csr_array, csr_array):
    if verbose:
        print("Computing weights...")
        start_time = time.perf_counter()
    if phi <= 0:
        raise ValueError("phi must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    tree = KDTree(X.T)
    dist, k_nearest = tree.query(X.T, k=k + 1, workers=-1)
    N = X.shape[1]
    weight_matrix = lil_array((N, N))
    for i in range(N):
        neighbors = k_nearest[i, : k + 1]
        distances = dist[i, : k + 1]
        weight_values = np.exp(-phi * distances**2)
        weight_matrix[i, neighbors] = weight_values
        weight_matrix[neighbors, i] = weight_values
    weight_matrix.setdiag(0)
    weight_matrix = weight_matrix.tocsr()
    weight_matrix *= gamma
    idx_r, idx_c, val = find(triu(weight_matrix))
    num_weight = len(val)
    W = lil_array((N, num_weight))
    Wbar = lil_array((N, num_weight))
    # Set 1 at specified positions
    W[idx_r, np.arange(num_weight)] = 1
    Wbar[idx_c, np.arange(num_weight)] = 1
    W = W.tocsr()
    Wbar = Wbar.tocsr()
    weight_vec = val.T
    node_arc_matrix = W - Wbar
    if verbose:
        end_time = time.perf_counter()
        print("Weights computed in {} seconds.".format(end_time - start_time))
    return weight_vec, node_arc_matrix, weight_matrix, end_time - start_time
