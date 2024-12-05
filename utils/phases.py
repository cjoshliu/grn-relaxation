# phases.py: functions for characterizing phase transitions


import jax
from jax import numpy as jnp
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem

from utils.modelIO import *


def get_ac_time(xs: jnp.ndarray, N: int) -> float:
    '''Compute autocorrelation time.

    Parameters
    ----------
    xs : jnp.ndarray, 2D, int
        Nonnegative integer GRN replica states over N_steps

    N : int
        Number of genes
    
    Returns
    -------
    t_corr : float
        Correlation time
    
    t_corr_err : float
        Standard error of correlation time
    '''
    # get log of minimum-subtracted autocorrelation for exponential fit
    C = get_avg_autocorrelation(xs, N)
    t = jnp.arange(xs.shape[1])

    # define exponential decay for optimization
    def exp_decay(t, a, t_corr, c):
        return c+a*jnp.exp(-t/t_corr)

    popt, pcov = curve_fit(exp_decay, t, C) # fit curve
    return jnp.array(popt[1]), jnp.sqrt(pcov[1, 1])


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


def get_autocorrelation(grn_states: jnp.ndarray, N: int) -> float:
    '''Estimate autocorrelation from single trajectory.

    Parameters
    ----------
    grn_states : jnp.ndarray, 1D, int
        Nonnegative integer GRN states over N_steps
    
    N : int
        Nonnegative integer number of genes
    
    Returns
    -------
    C : jnp.ndarray, 1D, float
        Autocorrelation averaged over genes
    '''
    # function to encode batch of GRN states
    def encode_instance(grn_state):
        return encode_grn_state(grn_state, N)
    encode_batch = jax.vmap(encode_instance)
    grn_state_codes = encode_batch(grn_states)
    C = 2.0/N*(grn_state_codes == grn_state_codes[0]).sum(axis=1)-1.0
    return C


def get_avg_autocorrelation(xs: jnp.ndarray, N: int) -> jnp.ndarray:
    '''Compute average autocorrelation over multiple trajectories.

    Parameters
    ----------
    xs : jnp.ndarray, 2D, int
        Nonnegative integer GRN replica states over N_steps

    N : int
        Number of genes
    
    Returns
    -------
    C_avg : jnp.ndarray, 1D, float
        Autocorrelation averaged over replicas
    '''
    def get_autocorrelation_instance(x):
        return get_autocorrelation(x, N)
    get_autocorrelation_optimized = jax.vmap(get_autocorrelation_instance)
    return jnp.mean(get_autocorrelation_optimized(xs), axis=0)


# def get_magnetization(grn_state: int, N: int) -> float:
#     '''Estimate magnetization for a single GRN state.
    
#     Parameters
#     ----------
#     grn_state : int
#         Nonnegative integer GRN state

#     N : int
#         Nonnegative integer number of genes

#     Returns
#     -------
#     m : float
#         Average per-gene magnetization
#     '''
#     grn_state_code = encode_grn_state(grn_state, N)
#     return 2.0*grn_state_code.sum()/N-1.0


# def get_avg_magnetization(grn_states: int, N: int):
#     '''Estimate average magnetization for multiple GRN states.
    
#     Parameters
#     ----------
#     grn_states : jnp.ndarray, 1D, int
#         Nonnegative integer representations of GRN states

#     N : int
#         Number of genes

#     Returns
#     -------
#     m_mu : float
#         Mean per-gene magnetization

#     m_se : float
#         Standard error of mean per-gene magnetization
#     '''
#     get_m_vmap = jax.vmap(lambda grn_state: get_magnetization(grn_state, N))
#     ms = get_m_vmap(grn_states)
#     return jnp.mean(ms), jnp.array(sem(ms))


# def get_dynamic_exp(xs: jnp.ndarray, N: int):
#     '''Compute dynamic exponent.

#     Parameters
#     ----------
#     xs : jnp.ndarray, 2D, int
#         Nonnegative integer GRN replica states over N_steps

#     N : int
#         Number of genes
    
#     Returns
#     -------
#     vz : float
#         Dynamic exponent estimate
    
#     se_vz : float
#         Standard error of dynamic-exponent estimate

#     r2 : float
#         Coefficient of determination of linear fit
#     '''
#     C = get_avg_autocorrelation(xs, N)
#     t = jnp.arange(C.shape[0])

#     nan_inf_mask  = (C > 0.0) & (t > 0.0)
#     neg_vz, _, r, _, se_vz = linregress(jnp.log(t[nan_inf_mask]), jnp.log(C[nan_inf_mask]))
#     return jnp.array(-neg_vz), jnp.array(se_vz), jnp.array(r**2)


# def get_qAE(grn_states: jnp.ndarray, N: int, Ttr: int) -> float:
#     '''Estimate Edwards-Anderson overlap from single trajectory.
    
#     Parameters
#     ----------
#     grn_states : jnp.ndarray, 1D, int
#         Nonnegative integer GRN states over N_steps
    
#     N : int
#         Nonnegative integer number of genes

#     Ttr : int
#         Initial transient timesteps to discard
    
#     Returns
#     -------
#     qAE : jnp.ndarray, 0D, float
#         Edwards-Anderson overlap
#     '''
#     N_steps = grn_states.shape[0]-Ttr-1 # get time interval

#     # function to encode batch of GRN states
#     def encode_instance(grn_state):
#         return encode_grn_state(grn_state, N)
#     encode_batch = jax.vmap(encode_instance)

#     # compute autocorrelations
#     on_steps = encode_batch(grn_states[Ttr+1:]).sum(axis=0)
#     mean_spins  = 2.0*on_steps/N_steps-1.0
#     initial_spins = jnp.where(encode_instance(grn_states[Ttr]), 1.0, -1.0)
#     return jnp.dot(initial_spins, mean_spins)/N
