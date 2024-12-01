# sim.py: functions and classes for simulating GRN evolution


from jax import numpy as jnp
from jax import lax
from jax import random

from criteria import *
from grn import *
from tsl import *


def mcmc_step(key: random.PRNGKey,
              grn_state: int,
              grn: jnp.ndarray,
              T: float,
              criterion: AcceptanceCriterion,
              N: int):
    '''Perform one Monte-Carlo time step.

    Parameters
    ----------
    key : random.PRNGKey
        Randomization key
    
    grn_state : jnp.ndarray, 0D, int
        Nonnegative integer representation of grn state

    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    T : jnp.ndarray, 0D, float
        Pseudo-temperature

    criterion : AcceptanceCriterion
        Markov-Chain Monte-Carlo acceptance criterion

    N : jnp.ndarray, 0D, int
        Nonnegative integer number of genes
    
    Returns
    -------
    subkey : random.PRNGKey
        Fresh randomization key for next step

    updated_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state after MCMC step
    '''
    updated_state_code = encode_grn_state(grn_state, N) # initialize state code
    updated_energy = get_grn_energy(grn, updated_state_code) # initialize energy
    subkey = key # initialize randomization key

    for i in range(N): # propose and accept or reject N flips
        gid_to_flip = random.randint(subkey, (), 0, N) # propose random gene to flip
        _, subkey = random.split(subkey) # refresh key for stochastic acceptance
        # get eenrgy and state code after proposed gene flip
        proposed_energy, proposed_state_code = get_proposal(grn, updated_state_code, gid_to_flip)
        # randomly accept or reject proposal
        accept = random.uniform(subkey) <= criterion(updated_energy, proposed_energy, T)
        updated_energy = (~accept)*updated_energy+accept*proposed_energy
        updated_state_code = (~accept)*updated_state_code+accept*proposed_state_code
        _, subkey = random.split(subkey) # refresh key for next proposal

    return subkey, decode_grn_state(updated_state_code)


def mcmc_trajectory(key, grn_state, grn, T, criterion, N, N_steps):
    """
    Evolve a single GRN state over N_steps using mcmc_step.

    Parameters
    ----------
    key : random.PRNGKey
        Randomization key
    
    grn_state : jnp.ndarray, 0D, int
        Nonnegative integer representation of grn state

    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    T : jnp.ndarray, 0D, float
        Pseudo-temperature

    criterion : AcceptanceCriterion
        Markov-Chain Monte-Carlo acceptance criterion

    N : jnp.ndarray, 0D, int
        Nonnegative integer number of genes
    
    N_steps : jnp.ndarray, 0D, int
        Number of steps to evolve GRN

    Returns
    -------
    grn_trajectory : jnp.ndarray, 1D, int
        Nonnegative integer GRN states over N_steps
    """
    def step_fn(carry, _):
        current_key, current_state = carry
        # Perform one MCMC step
        next_key, next_state = mcmc_step(current_key, current_state, grn, T, criterion, N)
        return (next_key, next_state), next_state

    # Run the scan loop
    initial_carry = (key, grn_state)
    _, grn_trajectory = lax.scan(step_fn, initial_carry, jnp.arange(N_steps))
    return grn_trajectory
