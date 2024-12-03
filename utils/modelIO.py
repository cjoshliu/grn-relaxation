# modelIO.py: functions for initializing MCMC experiments


from jax import numpy as jnp
from jax import random
import pandas as pd


def load_grn(grn_path: str, gids_path: str):
    '''Load GRN data.

    Parameters
    ----------
    grn_path : str
        Path to .npy file containing grn data
    
    gid_path : str
        Path to .csv file containing gene names and indices

    Returns
    -------
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    gids : dict
        Keys are gene names, values are gene indices

    N : int
        Nonnegative integer number of genes
    '''
    grn = jnp.load(grn_path)
    gids_df = pd.read_csv(gids_path)
    gids = dict(zip(gids_df['gene'], gids_df['id']))
    N = len(gids_df)
    return grn, gids, N


def encode_grn_state(grn_state: int, N: int) -> jnp.ndarray:
    '''Encode GRN state to bool array.

    Parameters
    ----------
    grn_state : jnp.ndarray, 0D, int
        Nonnegative integer representation of grn state

    N : int
        Nonnegative integer number of genes
    
    Returns
    -------
    grn_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state
    '''
    bit_positions = jnp.arange(N)  # Little-endian bit order
    return (grn_state & (1 << bit_positions)) > 0


def decode_grn_state(grn_state_code: jnp.ndarray) -> int:
    '''Decode GRN state from bool array.

    Parameters
    ----------
    grn_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state
    
    Returns
    -------
    grn_state : jnp.ndarray, 0D, int
        Nonnegative integer representation of GRN state
    '''
    # powers of 2 in little-endian order
    powers_of_2 = 2**jnp.arange(grn_state_code.shape[0])
    return jnp.dot(powers_of_2, grn_state_code).astype(jnp.int32)


def randomize_grn(key: random.PRNGKey, grn: jnp.ndarray) -> jnp.ndarray:
    '''Create random GRN with same degree distribution as grn.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Randomization key
    
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    Returns
    -------
    rgrn : jnp.ndarray, same shape as grn
        Random GRN with same degree distribution as grn
    '''
    # shuffle target-gene indices
    shuffled_idcs = random.permutation(key, grn.shape[1])
    shuffled_targets = grn[1, shuffled_idcs]
    # return grn copy with shuffled target genes
    return grn.at[1].set(shuffled_targets)
