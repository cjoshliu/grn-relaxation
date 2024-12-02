# sim.py: functions and classes for simulating GRN evolution


from jax import numpy as jnp
from jax import lax
from jax import random

from criteria import *
from energies import *
from modelIO import *


def get_proposal(grn: jnp.ndarray,
                 grn_state_code: jnp.ndarray,
                 field: jnp.ndarray,
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
    
    field : jnp.ndarray, 1D, float
        Field strength at each gene
    
    gid_to_flip : jnp.ndarray, 0D, int
        ID of gene proposed for flipping

    Returns
    -------
    proposed_energy : jnp.ndarray, 0D, int
        Pseudo-Hamiltonian of proposed GRN state
    
    proposed_state_code : jnp.ndarray, 1D, bool
        Binary representation of proposed GRN state
    '''
    # get grn state code and energy change after gene flip
    proposed_state_code = grn_state_code.at[gid_to_flip].set(~grn_state_code[gid_to_flip])
    proposed_energy = get_hamiltonian(grn, proposed_state_code, field)
    return proposed_energy, proposed_state_code


def get_step(key: random.PRNGKey,
             grn_state: int,
             grn: jnp.ndarray,
             field : jnp.ndarray,
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
    
    field : jnp.ndarray, 1D, float
        Field strength at each gene

    T : jnp.ndarray, 0D, float
        Pseudo-temperature

    criterion : AcceptanceCriterion
        Markov-Chain Monte-Carlo acceptance criterion

    N : int
        Nonnegative integer number of genes
    
    Returns
    -------
    subkey : random.PRNGKey
        Fresh randomization key for next step

    updated_state_code : jnp.ndarray, 1D, bool
        Binary representation of GRN state after MCMC step
    
    velocity : jnp.ndarray, 0D, int
        Number of accepted flips
    '''
    updated_state_code = encode_grn_state(grn_state, N) # initialize state code
    updated_energy = get_hamiltonian(grn, updated_state_code, field) # initialize energy
    subkey = key # initialize randomization key

    def step_fn(carry, _):
        subkey, updated_state_code, updated_energy = carry
        # Propose random gene to flip
        gid_to_flip = random.randint(subkey, (), minval=0, maxval=N)
        # Split keys for randomness
        _, subkey = random.split(subkey)
        # Get proposed energy and state code after gene flip
        proposed_energy, proposed_state_code = get_proposal(grn, updated_state_code, field, gid_to_flip)
        # Decide whether to accept the proposal
        accept = random.uniform(subkey) <= criterion(updated_energy, proposed_energy, T)

        # Update energy and state code based on acceptance
        updated_energy = jnp.where(accept, proposed_energy, updated_energy)
        updated_state_code = jnp.where(accept, proposed_state_code, updated_state_code)
        _, subkey = random.split(subkey) # refresh key for next proposal
        new_carry = (subkey, updated_state_code, updated_energy) # Return updated carry
        return new_carry, accept.astype(int)

    # Initial carry for the scan
    initial_carry = (subkey, updated_state_code, updated_energy)
    # Run the scan over N iterations
    (subkey, updated_state_code, updated_energy), accepts = lax.scan(
        step_fn, initial_carry, xs=None, length=N
    )

    return subkey, decode_grn_state(updated_state_code), accepts.sum()


def get_trajectory(key: random.PRNGKey,
                   grn_state: int,
                   grn: jnp.ndarray,
                   field : jnp.ndarray,
                   T: float,
                   criterion: AcceptanceCriterion,
                   N: int,
                   N_steps: int) -> jnp.ndarray:
    """Evolve a single GRN state over N_steps steps and return full trajectory.

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
    
    field : jnp.ndarray, 1D, float
        Field strength at each gene

    T : float
        Pseudo-temperature

    criterion : AcceptanceCriterion
        Markov-Chain Monte-Carlo acceptance criterion

    N : int
        Nonnegative integer number of genes
    
    N_steps : int
        Number of steps to evolve GRN

    Returns
    -------
    grn_states : jnp.ndarray, 1D, int
        Nonnegative integer GRN states over N_steps
        Note that the zeroth (initial) state is not included

    grn_velocities : jnp.ndarray, 1D, int
        Number of accepted flips over N_steps

    """
    def step_fn(carry, _):
        current_key, current_state = carry
        # Perform one MCMC step
        next_key, next_state, velocity = get_step(current_key,
                                                  current_state,
                                                  grn,
                                                  field,
                                                  T,
                                                  criterion,
                                                  N)
        new_carry = (next_key, next_state)
        outputs = (next_state, velocity)
        return new_carry, outputs

    # Run the scan loop
    initial_carry = (key, grn_state)
    (_, _), (grn_states, grn_velocities) = lax.scan(step_fn, initial_carry, jnp.arange(N_steps))
    grn_states = jnp.concatenate((jnp.array([grn_state]), grn_states), axis=0)
    return grn_states, grn_velocities


def get_final_state(key: random.PRNGKey,
                    grn_state: int,
                    grn: jnp.ndarray,
                    field: jnp.ndarray,
                    T: float,
                    criterion: AcceptanceCriterion,
                    N: int,
                    N_steps: int):
    """Evolve a single GRN state over N_steps steps and return final state.

    Parameters
    ----------
    key : random.PRNGKey
        Randomization key
    
    grn_state : int
        Nonnegative integer representation of GRN state

    grn : jnp.ndarray, 2D, int
        Row 0 contains IDs of regulator genes
        Row 1 contains IDs of target genes
        Row 2 contains interactions, 0 for inhibition, 1 for activation

    field : jnp.ndarray, 1D, float
        Field strength at each gene

    T : float
        Pseudo-temperature

    criterion : AcceptanceCriterion
        MCMC acceptance criterion

    N : int
        Number of genes
    
    N_steps : int
        Number of steps to evolve GRN

    Returns
    -------
    final_state : int
        Nonnegative integer GRN state after N_steps
    """
    def step_fn(carry, _):
        current_key, current_state = carry
        # Perform one MCMC step
        next_key, next_state, _ = get_step(
            current_key, current_state, grn, field, T, criterion, N
        )
        return (next_key, next_state), None  # Return updated carry

    # Initial carry
    initial_carry = (key, grn_state)
    # Run the scan loop
    final_carry, _ = lax.scan(
        step_fn, initial_carry, xs=None, length=N_steps
    )
    # Extract final state
    _, final_state = final_carry

    return final_state
