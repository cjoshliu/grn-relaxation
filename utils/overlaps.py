# overlaps.py: functions for estimating glass order parameters


import jax
from jax import numpy as jnp

from utils.modelIO import *


def get_qAE(grn_states: jnp.ndarray, N: int, Ttr: int) -> float:
    '''Estimate Edwards-Anderson overlap from single trajectory.
    
    Parameters
    ----------
    grn_states : jnp.ndarray, 1D, int
        Nonnegative integer GRN states over N_steps
    
    N : int
        Nonnegative integer number of genes

    Ttr : int
        Initial transient timesteps to discard
    
    Returns
    -------
    qAE : jnp.ndarray, 0D, float
        Edwards-Anderson overlap
    '''
    N_steps = grn_states.shape[0]-Ttr-1 # get time interval

    # function to encode batch of GRN states
    def encode_instance(grn_state):
        return encode_grn_state(grn_state, N)
    encode_batch = jax.vmap(encode_instance)

    # compute autocorrelations
    on_steps = encode_batch(grn_states[Ttr+1:]).sum(axis=0)
    mean_spins  = 2.0*on_steps/N_steps-1.0
    initial_spins = jnp.where(encode_instance(grn_states[Ttr]), 1.0, -1.0)
    return jnp.dot(initial_spins, mean_spins)/N


def get_qab(grn_state_a: int, grn_state_b: int, N: int) -> float:
    '''Estimate overlap of two replicas a and b.

    Parameters
    ----------
    grn_state_a : jnp.ndarray, 0D, int
        Nonnegative integer GRN state a
    
    grn_state_b : jnp.ndarray, 0D, int
        Nonnegative integer GRN state b
    
    N : int
        Nonnegative integer number of genes
    
    Returns
    -------
    qab : jnp.ndarray, 0D, float
        Replica overlap
    '''
    grn_code_a = encode_grn_state(grn_state_a, N)
    grn_code_b = encode_grn_state(grn_state_b, N)
    equal_spins = (grn_code_a == grn_code_b).sum()
    return 2.0*equal_spins/N-1.0


def get_all_qab(replicas: jnp.ndarray, N: int) -> jnp.ndarray:
    '''Compute overlaps between all unordered pairs of replicas.

    Parameters
    ----------
    grn_states : jnp.ndarray, 1D, int
        Nonnegative integer representations of GRN states

    N : int
        Number of genes

    Returns
    -------
    qab_values : jnp.ndarray, 1D, float
        Overlaps between all unordered pairs of replicas, excluding self-overlaps
    '''
    num_replicas = replicas.shape[0]

    # Get indices of all unordered pairs (i, j) where i < j
    idx_i, idx_j = jnp.triu_indices(num_replicas, k=1)

    # Extract the GRN states for these indices
    grn_state_a = replicas[idx_i]
    grn_state_b = replicas[idx_j]

    # Vectorize get_qab over grn_state_a and grn_state_b
    vectorized_get_qab = jax.vmap(get_qab, in_axes=(0, 0, None))

    # Compute the overlaps
    qab_values = vectorized_get_qab(grn_state_a, grn_state_b, N)

    return qab_values
