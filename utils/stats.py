# stats.py: miscellaneous GRN simulation statistics


import jax
from jax import numpy as jnp

from utils.modelIO import *


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
