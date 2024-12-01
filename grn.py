# grn.py: functions for manipulating gene regulatory network


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

    gids : pd.DataFrame
        Column 0 contains gene names
        Column 1 contains gene IDs
    '''
    grn = jnp.load(grn_path)
    gids = pd.read_csv(gids_path)
    return grn, gids


def encode_grn_state(grn_state: int, N: int) -> jnp.ndarray:
    '''Encode GRN state to bool array.

    Parameters
    ----------
    grn_state : jnp.ndarray, 0D, int
        Nonnegative integer representation of grn state

    N : jnp.ndarray, 0D, int
        Nonnegative integer number of genes
    
    Returns
    -------
    grn_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state
    '''

    # if not 0 <= grn_state < 2**N:
    #     raise ValueError('grn_state must be in [0, 2**N)')
    bit_positions = jnp.arange(N-1, -1, -1) # big-endian bit order
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
    # if not (grn_state_code.dtype == bool and jnp.ndim(grn_state_code)):
    #     raise ValueError('grn_state_code must be 1D and have dtype bool')
    # powers of 2 in big-endian order
    powers_of_2 = 2**jnp.arange(grn_state_code.shape[0]-1, -1, -1)
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
    # if not grn.shape[0] == 3:
    #     raise ValueError('grn must have exactly 3 rows.')
    # shuffle target-gene indices
    shuffled_idcs = random.permutation(key, grn.shape[1])
    shuffled_targets = grn[1, shuffled_idcs]
    # return grn copy with shuffled target genes
    return grn.at[1].set(shuffled_targets)


def get_grn_energy(grn: jnp.ndarray, grn_state_code: jnp.ndarray) -> int:
    '''Evaluate pseudo-Hamiltonian of GRN state.

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
    grn_energy : jnp.ndarray, 0D, int
        Pseudo-Hamiltonian of GRN state
    '''
    # if not grn.shape[0] == 3:
    #     raise ValueError('grn must have exactly 3 rows.')
    # evaluate spin products
    summands = grn_state_code[grn[0]] == grn_state_code[grn[1]]
    summands = summands == grn[2].astype(bool) # multiply by couplings
    return summands.shape[0]-2*jnp.sum(summands.astype(int))


def get_proposal(grn: jnp.ndarray,
                 grn_state_code: jnp.ndarray,
                 gid_to_flip: int):
    '''Evaluate change in pseudo-Hamiltonian from proposed gene flip.

    Parameters
    ----------
    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    grn_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state
    
    gid_to_flip : jnp.ndarray, 0D, int
        ID of gene proposed for flipping

    Returns
    -------
    proposed_energy : jnp.ndarray, 0D, int
        Pseudo-Hamiltonian of proposed GRN state
    
    proposed_state_code : jnp.ndarray, 1D, bool
        Binary representation of proposed GRN state
    '''
    # if not grn.shape[0] == 3:
    #     raise ValueError('grn must have exactly 3 rows.')
    # if not gid_to_flip < grn_state_code.shape[0]:
    #     raise ValueError('gid_to_flip is out of range')
    # get grn state code and energy change after gene flip
    proposed_state_code = grn_state_code.at[gid_to_flip].set(~grn_state_code[gid_to_flip])
    proposed_energy = get_grn_energy(grn, proposed_state_code)
    return proposed_energy, proposed_state_code
