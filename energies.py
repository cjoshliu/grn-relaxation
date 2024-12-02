# energies.py: functions for estimating GRN energies


import jax
from jax import numpy as jnp

from modelIO import *


def get_pct_frustration(grn: jnp.ndarray, grn_state: int, N: int) -> float:
    '''Evaluate percent frustration of GRN state.

    Parameters
    ----------
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation
    
    grn_state : jnp.ndarray, 0D, int
        Nonnegative integer representation of GRN state
    
    N : int
        Nonnegative integer number of genes

    Returns
    -------
    pct_frustration : jnp.ndarray, 0D, float
        Percent frustration of GRN state
    '''
    grn_state_code = encode_grn_state(grn_state, N) # get binary representation
    satisfied_edges = get_satisfied_edges(grn, grn_state_code) # get satisfied edges
    return (grn.shape[1]-satisfied_edges)/grn.shape[1]


def get_free_energy(grn: jnp.ndarray, field: jnp.ndarray, T: float) -> jnp.ndarray:
    '''Get free energy of GRN.

    Parameters
    ----------
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    field : jnp.ndarray, 1D, float
        Field strength at each gene

    T : float
        Pseudo-temperature

    Returns
    -------
    free_energy : jnp.ndarray, 0D, float
        Quenched free energy
    '''
    # get array of possible states
    N = field.shape[0]
    grn_states = jnp.arange(2**N)

    # get pseudo-Hamiltonians of all states
    def get_instance_energy(grn_state):
        grn_state_code = encode_grn_state(grn_state, N)
        return get_hamiltonian(grn, grn_state_code, field)
    get_instance_energy_vmap = jax.vmap(get_instance_energy)

    # get partition function
    Z = jnp.exp(-get_instance_energy_vmap(grn_states)/T).sum()
    return -T*jnp.log(Z)


### HELPERS ###


def get_satisfied_edges(grn: jnp.ndarray, grn_state_code: jnp.ndarray) -> int:
    '''Count satisfied edges in a GRN.

    Parameters
    ----------
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation
    
    grn_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state

    Returns
    -------
    satisfied_edges : jnp.ndarray, 0D, int
        Number of satisfied edges
    '''
    # get spin products
    summands = grn_state_code[grn[0]] == grn_state_code[grn[1]]
    summands = summands == grn[2].astype(bool) # multiply by couplings
    return jnp.sum(summands)


def get_hamiltonian(grn: jnp.ndarray,
                   grn_state_code: jnp.ndarray,
                   field: jnp.ndarray) -> float:
    '''Evaluate pseudo-Hamiltonian of GRN state.

    Parameters
    ----------
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation
    
    grn_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state
    
    field : jnp.ndarray, 1D, float
        Field strength at each gene
    
    Returns
    -------
    grn_energy : jnp.ndarray, 0D, float
        Pseudo-Hamiltonian of GRN state
    '''
    # get energy from couplings
    energy_J = grn.shape[1]-2*get_satisfied_edges(grn, grn_state_code)
    # get energy from external field
    energy_h = jnp.dot(jnp.where(grn_state_code, 1.0, -1.0), field)
    return energy_J-energy_h