# tsl.py: functions for calculating topological speed limits


import jax
from jax import lax
import jax.numpy as jnp
import jax.tree_util as jtu
from ott.geometry import geometry
# from ott.geometry.costs import TICost, PNormP
# from ott.geometry.pointcloud import PointCloud
from ott.solvers import linear

from utils.modelIO import encode_grn_state


def get_bit_cost_matrix(size: int) -> jnp.ndarray:
    '''Compute the bit distance matrix for integers from 1 to N.

    Parameters
    ----------
    size : int
        The number of integers.

    Returns
    -------
    cost : jnp.ndarray, 2D, int
        An NxN matrix where cost[i, j] is the number of bits that differ between i and j.
    '''
    # Create an array of integers from 1 to N
    x = jnp.arange(size, dtype=jnp.int32)  # Shape: (N,)

    # Compute pairwise XOR between all integers
    # This creates an NxN matrix where each element is x[i] ^ x[j]
    xor_matrix = x[:, None] ^ x[None, :]  # Shape: (N, N)

    # Compute the Hamming distance using population count
    cost = lax.population_count(xor_matrix)  # Shape: (N, N)

    return cost.astype(jnp.float32)


def get_wasserstein1(geom, a, b) -> float:
    '''L1 Wasserstein distance in 1D.

    Parameters
    ----------
    a : jnp.ndarray, 1D, float
        Source histogram, elements sum to 1
    
    b : jnp.ndarray, 1D, float
        Target histogram, elements sum to 1

    Returns
    -------
    w1 : float
        L1 Wasserstein distance between a and b
    '''
    assert a.shape == b.shape
    sol = linear.solve(geom, a=a, b=b)
    return sol.primal_cost


def get_histogram(grn_states: jnp.ndarray, N: int):
    '''Compute histogram of GRN states.

    Parameters
    ----------
    grn_states : jnp.ndarray, 1D, int
        Nonnegative integer representations of GRN states

    N : int
        Number of genes

    Returns
    -------
    hist : jnp.ndarray, 1D, float
        Histogram of GRN states
    '''
    cts = jnp.bincount(grn_states, minlength=2**N, length=2**N)
    return cts/grn_states.shape[0]


def get_all_w1(xs: jnp.ndarray, N: int) -> jnp.ndarray:
    '''Compute Wasserstein distances between histograms of GRN states over time.

    Parameters
    ----------
    xs : jnp.ndarray, 2D, int
        Array of GRN states over time. Each column represents GRN states at a point in time.
    
    N : int
        Number of genes

    Returns
    -------
    w1_dists : jnp.ndarray, 1D, float
        Array of Wasserstein distances between the histogram at each time point and the initial histogram.
    '''
    # Compute the initial histogram
    hist0 = get_histogram(xs.T[0], N)  # xsT[0] is a 1D array
    # Vectorize get_histogram over the columns of xs
    get_histogram_vmap = jax.vmap(get_histogram, in_axes=(0, None))
    # Compute histograms for all time points
    hists = get_histogram_vmap(xs.T[1:], N)  # Shape: (N_steps, 2^N)

    # Define optimal transport geometry
    cost = get_bit_cost_matrix(2**N)
    geom = geometry.Geometry(cost_matrix=cost)

    # Define a vectorized function to compute Wasserstein distances to hist0
    compute_wasserstein_vmap = jax.vmap(
        lambda hist_i: get_wasserstein1(geom, hist0, hist_i)
    )
    # Compute Wasserstein distances for each histogram
    w_dists = compute_wasserstein_vmap(hists)  # Shape: (num_time_points,)

    return w_dists*xs.shape[1]


# Poorly optimized estimator of the L1 Wasserstein distance for testing
# This gives different results from wasserstein1 when using BitDifference
# wasserstein1 results are analytically correct
# Sinkhorn regularization seems to have large impact when using BitDifference
# def wasserstein1_naive(a, b, cost_fn=PNormP(1), min_iterations=0, max_iterations=100):
#     '''L1 Wasserstein distance.'''
#     assert a.shape == b.shape
#     assert len(a.shape) == 1
#     x = jnp.arange(a.shape[0]).astype(jnp.float32)[:, jnp.newaxis]
#     geom = PointCloud(x, x, cost_fn=cost_fn)
#     sol = linear.solve(geom, a=a, b=b,
#                        lse_mode=False,
#                        min_iterations=min_iterations,
#                        max_iterations=max_iterations)
#     return sol.primal_cost


# def bits_in_difference(z):
#     '''Number of bits used to encode an integer.'''
#     return lax.population_count(jnp.abs(z).astype(jnp.int32)).reshape(()).astype(jnp.float32)


# @jtu.register_pytree_node_class
# class BitDifference(TICost):
#     '''Number of differing bits'''
#     def __init__(self):
#         super().__init__()
    
#     def h(self, z: jnp.ndarray) -> float:
#         cand_0 = bits_in_difference(z)
#         complement = 2**(jnp.ceil(jnp.log2(jnp.abs(z))+1))-jnp.abs(z)
#         cand_1 = bits_in_difference(complement)
#         return jnp.stack((cand_0, cand_1)).min()


# def wasserstein1(a, b, cost_fn=BitDifference()) -> float:
#     '''L1 Wasserstein distance in 1D.

#     Parameters
#     ----------
#     a : jnp.ndarray, 1D, float
#         Source histogram, elements sum to 1
    
#     b : jnp.ndarray, 1D, float
#         Target histogram, elements sum to 1

#     cost_fn : ott.geometry.costs.TICost, optional
#         Translation-invariant cost function
#         Defaults to BitDifference() to compute Wasserstein distances between Ising configurations
#         Use PNormP(1) for usual L1 norm

#     Returns
#     -------
#     w1 : float
#         L1 Wasserstein distance between a and b
#     '''
#     if not a.shape == b.shape: raise ValueError('a and b must have the same shape')
#     if not len(a.shape) == 1: raise ValueError('a and b must be 1D')
#     x = jnp.arange(a.shape[0])[:, jnp.newaxis] # define discrete support [0, N_configs)
#     geom = PointCloud(x, x, cost_fn=cost_fn) # define optimization geometry
#     sol = linear.solve_univariate(geom, a=a, b=b) # solve for optimal transport plan
#     return sol.ot_costs.reshape(()) # return L1 Wasserstein distance