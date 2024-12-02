# overlaps.py: functions for estimating glass order parameters


import jax
from jax import numpy as jnp

from modelIO import *


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
