# stability.py: functions for analyzing gene regulatory networks


import jax
from jax import numpy as jnp

from utils.energies import *
from utils.mcmc import *
from utils.modelIO import *


def is_stable_state(grn: jnp.ndarray,
                    grn_state_code: jnp.ndarray,
                    field: jnp.ndarray) -> bool:
    '''Check whether grn_state is stable.

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
    stable : jnp.ndarray, 0D, bool
        Whether grn_state_code represents a stable state
    '''
    # create an array of gene IDs to flip
    gids_to_flip = jnp.arange(grn_state_code.shape[0])

    # define a function to get the proposed energy for a single gene flip
    def get_proposal_energy(gid_to_flip):
        proposed_energy, _ = get_proposal(grn, grn_state_code, field, gid_to_flip)
        return proposed_energy
    # vectorize the function over all gene IDs
    get_proposal_energy_vmap = jax.vmap(get_proposal_energy)

    # compute the energies of all neighboring states
    neighbor_energies = get_proposal_energy_vmap(gids_to_flip)
    # compute the current energy
    current_energy = get_hamiltonian(grn, grn_state_code, field)
    # check if the current energy is less than or equal to all neighbor energies
    stable = jnp.all(current_energy <= neighbor_energies)

    return stable


def get_stable_states(grn: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
    '''Get stable states of GRN.

    Parameters
    ----------
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    field : jnp.ndarray, 1D, float
        Field strength at each gene

    Returns
    -------
    stable_states : jnp.ndarray, 1D, int
    '''
    # get array of possible states
    N = field.shape[0]
    grn_states = jnp.arange(2**N)

    # define a function to get stability of one state
    def get_stability(grn_state):
        grn_state_code = encode_grn_state(grn_state, N)
        return is_stable_state(grn, grn_state_code, field)
    # vectorize the function over all states
    get_stability_vmap = jax.vmap(get_stability)

    stabilities = get_stability_vmap(grn_states)
    return grn_states[stabilities]


def get_expression_level(grn_states: jnp.ndarray, gids: dict, Ttr: int) -> float:
    '''Get time-averaged expression level of a gene.
    
    Parameters
    ----------
    grn_states : jnp.ndarray, 1D, int
        Nonnegative integer GRN states over trajectory

    gids : dict
        Keys are gene names, values are gene indices
    
    gene : str
        Name of gene
    
    Ttr : int
        Initial transient timesteps to discard
    
    Returns
    -------
    elevels : dict
        Keys are gene names, values are expression levels
    '''
    # get number of genes and steps
    N = len(gids)
    N_steps = grn_states.shape[0]-Ttr

    # function to encode batch of GRN states
    N = len(gids)
    def encode_instance(grn_state):
        return encode_grn_state(grn_state, N)
    encode_batch = jax.vmap(encode_instance)

    # compute autocorrelations
    on_time = encode_batch(grn_states[Ttr:]).sum(axis=0)/N_steps
    return {gene_name: on_time[gene_id] for gene_name, gene_id in gids.items()}
