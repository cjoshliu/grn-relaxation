# tsl.py: functions for calculating topological speed limits


from jax import lax
import jax.numpy as jnp
import jax.tree_util as jtu
from ott.geometry.costs import TICost, PNormP
from ott.geometry.pointcloud import PointCloud
from ott.solvers import linear


@jtu.register_pytree_node_class
class BitDifference(TICost):
    '''Number of differing bits'''
    def __init__(self):
        super().__init__()
    
    def h(self, z: jnp.ndarray) -> float:
        return lax.population_count(jnp.abs(z).astype(jnp.int32)).reshape(()).astype(jnp.float32)


def wasserstein1(a, b, cost_fn=PNormP(1)):
    '''L1 Wasserstein distance in 1D.

    Parameters
    ----------
    a : jnp.ndarray, 1D, float
        Source histogram, elements sum to 1
    
    b : jnp.ndarray, 1D, float
        Target histogram, elements sum to 1

    cost_fn : ott.geometry.costs.TICost, optional
        Translation-invariant cost function
        Defaults to Manhattan distance PNormP(1)
        Use BitDifference() to compute Wasserstein distances between Ising configurations
    '''
    if not a.shape == b.shape: raise ValueError('a and b must have the same shape')
    if not len(a.shape) == 1: raise ValueError('a and b must be 1D')
    x = jnp.arange(a.shape[0])[:, jnp.newaxis] # define discrete support [0, N_configs)
    geom = PointCloud(x, x, cost_fn=cost_fn) # define optimization geometry
    sol = linear.solve_univariate(geom, a=a, b=b) # solve for optimal transport plan
    return sol.ot_costs.reshape(()) # return L1 Wasserstein distance


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